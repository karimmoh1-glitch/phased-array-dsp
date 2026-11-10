# phasedarray

A dependency-free Python DSP pipeline for **software-defined radio and acoustic phased arrays**. It takes
multichannel time-domain buffers (simulated, from a WAV file, or streamed block by block), transforms them
with a **from-scratch FFT** (no numpy, no scipy), cross-correlates the channels, and estimates the
**angle of arrival (AoA)** of the incoming signal.

- `fft.py`: iterative radix-2 Cooley–Tukey FFT, Bluestein chirp-z for any other length, packed real FFT
- `spectral.py`: Hann window, framing, sub-bin spectral peak estimation
- `correlation.py`: FFT cross-correlation and GCC-PHAT time-delay estimation with sub-sample interpolation
- `aoa.py`: linear-array geometry, plus narrowband (cross-spectral phase) and broadband (GCC-PHAT) AoA estimators
- `signals.py`: plane-wave simulator with mathematically exact fractional delays
- `stream.py`: block-streaming estimator for live audio / SDR buffers
- `audio_io.py`: 8/16/24/32-bit multichannel PCM WAV I/O using only the standard library
- `__main__.py`: command-line interface

The code is fully type-hinted and checks clean under `mypy --strict`.

## Quick start

Requires Python ≥ 3.10. The runtime has no dependencies; `pytest` and `mypy` are only needed for development.

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
```

```bash
.venv/bin/python -m phasedarray simulate --angle 37.5 --freq 1234.5
```

```bash
.venv/bin/python -m phasedarray simulate --angle -52 --method gcc_phat --snr 0
```

```bash
.venv/bin/python -m phasedarray wav recording.wav --spacing 0.05 --freq 1000
```

The WAV file must have one channel per array element, ordered along the array axis.

### Library use

```python
from phasedarray import LinearArray, estimate_aoa, read_wav, StreamingAoAEstimator

array = LinearArray.uniform(n_elements=4, spacing_m=0.1)           # acoustic, c = 343 m/s
fs, channels = read_wav("recording.wav")
est = estimate_aoa(channels, fs, array, method="narrowband", frequency_hz=1000.0)
print(est.angle_deg, est.delays_s, est.residual_rms_s)

# Live buffers of any block size, one estimate every frame_size/2 samples
stream = StreamingAoAEstimator(array, fs, frame_size=4096, frequency_hz=1000.0)
for block in audio_blocks():                                        # block = [ch0_samples, ch1_samples, ...]
    for est in stream.push(block):
        print(est.angle_deg)
```

For RF, pass `propagation_speed=SPEED_OF_LIGHT` and positions in metres. The estimator is agnostic
to the carrier, so it also works on a real IF signal.

## Visual assets

`assets/dsp_pipeline.svg` is the signal-flow diagram. Its generator is stdlib-only.
`assets/aoa_benchmark.png` is a 300 DPI accuracy plot. Every point in it comes from live runs of this
library, and it takes about 1.5 minutes to generate.

```bash
.venv/bin/python scripts/make_pipeline_svg.py
```

```bash
.venv/bin/pip install -e '.[plots]' && .venv/bin/python scripts/make_benchmark_png.py
```

## Verify and compile

```bash
.venv/bin/pytest
```

```bash
.venv/bin/mypy
```

```bash
.venv/bin/python -W error -m compileall -f src tests
```

---

## DSP equations

### 1. Discrete Fourier transform

For a length-N sequence x[n]:

$$X[k] = \sum_{n=0}^{N-1} x[n]\, W_N^{kn}, \qquad W_N = e^{-j2\pi/N}$$

$$x[n] = \frac{1}{N}\sum_{k=0}^{N-1} X[k]\, W_N^{-kn}$$

Computing this directly costs O(N²). Everything below brings it to O(N log N).

### 2. Radix-2 decimation-in-time FFT (`fft._radix2`)

When N is a power of two, split x into even- and odd-indexed halves E and O, each of length N/2, and
take their FFTs E[k] and O[k]. Because $W_N^{2kn} = W_{N/2}^{kn}$:

$$X[k] = E[k] + W_N^k\, O[k], \qquad X[k + N/2] = E[k] - W_N^k\, O[k], \qquad 0 \le k < N/2$$

That pair is the **butterfly**. Applying the split recursively gives log₂N stages of N/2 butterflies each.
The implementation is iterative and in place:

1. Permute the input into **bit-reversed order**, which is the order the recursion leaves the leaves in.
   The permutation table uses `rev[i] = (rev[i>>1] >> 1) | ((i & 1) << (bits-1))` and is cached per N.
2. For stage sizes 2, 4, …, N, apply the butterflies. The twiddle loop sits outside the block loop,
   so each $W$ is fetched once per stage.
3. Each twiddle factor $W_N^k = e^{-j2\pi k/N}$ is computed directly with `cmath.exp`. It is never
   built up by repeated multiplication, so twiddle error stays at about one ulp and does not grow with k.
   The tables are cached per (N, direction).

The inverse transform uses conjugated twiddles plus a 1/N scale.

### 3. Bluestein's chirp-z algorithm for any N (`fft._bluestein`)

Use the identity $kn = \tfrac{1}{2}\left[k^2 + n^2 - (k-n)^2\right]$ and the chirp $w[n] = e^{-j\pi n^2/N}$:

$$X[k] = w[k] \sum_{n=0}^{N-1} \big(x[n]\, w[n]\big)\, \overline{w[k-n]}$$

This is a **convolution** of $a[n] = x[n]w[n]$ with $b[n] = \overline{w[n]}$. Zero-pad both to a power
of two $M \ge 2N-1$ (b is wrapped so that negative indices land at M−n). The radix-2 FFT then evaluates
the convolution exactly. To keep the phase argument small for large n, the chirp is computed from
$n^2 \bmod 2N$, which is exact because $w$ has period 2N in $n^2$. Primes and other awkward lengths cost
O(N log N) this way.

### 4. Real-input FFT by packing (`fft.rfft`)

For real x of even length N, form the half-length complex sequence $z[n] = x[2n] + j\,x[2n+1]$ and take
its FFT Z, which has length M = N/2. The FFTs of the even and odd halves come back out through conjugate
symmetry:

$$E[k] = \tfrac12\big(Z[k] + \overline{Z[M-k]}\big), \qquad O[k] = \tfrac{1}{2j}\big(Z[k] - \overline{Z[M-k]}\big)$$

$$X[k] = E[k] + W_N^k\, O[k], \qquad k = 0 \dots N/2$$

This needs about half the arithmetic of a full complex FFT. `irfft` rebuilds the Hermitian spectrum
$X[N-k] = \overline{X[k]}$ and inverts it. If you ask it for more output samples than the original
length, the missing high bins count as zero, which is **band-limited (sinc) interpolation**.

### 5. Windowing and sub-bin frequency estimation (`spectral.py`)

The periodic Hann window is $w[n] = \tfrac12 - \tfrac12\cos(2\pi n/N)$. Its first sidelobe sits at
−31 dB and the sidelobes then roll off at 18 dB/octave. That suppresses leakage from the
negative-frequency image of a real tone, and 50%-overlapped frames sum to a constant (COLA).

`dominant_frequency` zero-pads by 4× and refines the peak bin k with **Gaussian (log-parabolic)
interpolation**. With α, β, γ the magnitudes at k−1, k and k+1:

$$\delta = \frac{1}{2}\,\frac{\ln\alpha - \ln\gamma}{\ln\alpha - 2\ln\beta + \ln\gamma}, \qquad \hat f = (k+\delta)\,\frac{f_s}{N_{\text{fft}}}$$

The Hann main lobe is close to Gaussian, so the error is well under 1% of a bin. The tests assert this.

### 6. Cross-correlation via the FFT (`correlation.cross_correlate`)

$$r_{ab}[\ell] = \sum_n a[n]\, b[n+\ell] \;\;\Longleftrightarrow\;\; R_{ab}[k] = \overline{A[k]}\, B[k]$$

The DFT computes a *circular* correlation. Zero-padding both signals to $N_{\text{fft}} \ge N_a + N_b - 1$
makes it equal to the linear one.

### 7. GCC-PHAT time-delay estimation (`correlation.gcc_phat`)

If $b(t) = a(t-D)$, then $B(f) = A(f)\,e^{-j2\pi fD}$. The generalised cross-correlation with the
**phase transform** whitens the cross-spectrum:

$$G[k] = \overline{A[k]}\,B[k], \qquad R_{\text{PHAT}}(\tau) = \mathcal{F}^{-1}\left\{\frac{G[k]}{|G[k]|}\right\} = \mathcal{F}^{-1}\left\{e^{-j2\pi f_k D}\right\} = \delta(\tau - D)$$

Only phase survives, so the peak is a sharp impulse at τ = D whatever the source's spectral colour,
and it is robust to reverberation. The implementation:

- **Hann-tapers** both records, which suppresses the edge discontinuity that PHAT would otherwise
  amplify in low-energy bins.
- Evaluates the inverse transform on a grid `interp` times finer (16× by default) by zero-padding the
  whitened spectrum, so the correlation is sinc-interpolated.
- Refines the peak with a **three-point parabolic fit** through $y_{-1}, y_0, y_{+1}$:
  $\delta = \tfrac12 (y_{-1} - y_{+1}) / (y_{-1} - 2y_0 + y_{+1})$.
- Limits the search to the physically possible lags $|\tau| \le |x_j - x_i|/c$ (plus two samples).

A positive delay means the second signal arrives later.

### 8. Array geometry and the plane-wave model (`aoa.LinearArray`)

Elements sit at positions $x_m$ on a line. A far-field source at angle θ from broadside, positive
towards +x, has wave vector direction $(\sin\theta, \cos\theta)$, so element m receives it at

$$\tau_m = -\frac{x_m \sin\theta}{c}$$

Here c is 343 m/s for air or 299 792 458 m/s for RF. The inter-element delay for a pair (i, j) is

$$D_{ij} = \tau_j - \tau_i = -\frac{(x_j - x_i)\sin\theta}{c}$$

### 9. Narrowband AoA from cross-spectral phase (`aoa.estimate_aoa_narrowband`)

For a tone at frequency f, a delay is a phase rotation. The steps:

1. Cut each channel into 50%-overlapped Hann frames and run `rfft` on each. Pick the bin
   $k = \text{round}(f N/f_s)$.
2. Average the **adjacent-pair cross-spectrum** over the frames (Welch). Noise averages down, the phase
   does not:
   $$C_m = \sum_{\text{frames}} \overline{X_{m-1}[k]}\, X_m[k], \qquad \psi_m = \arg C_m = \frac{2\pi f\,(x_m - x_{m-1})\sin\theta}{c}$$
3. **Unwrap across the aperture.** Each $\psi_m$ is unambiguous only if $|\psi_m| < \pi$, which means
   $d \le \lambda/2$ (the spatial Nyquist criterion). The cumulative sum
   $\Phi_m = \sum_{i \le m}\psi_i$ then gives the phase between element 0 and element m even when that
   span covers many wavelengths. Frequencies above $c / (2 d_{\max})$ are rejected with a
   spatial-aliasing error.
4. Convert to delays, $D_{0m} = -\Phi_m / (2\pi f)$, and fit (§11).

The negative-frequency image of the real tone is the only bias source, and the Hann window suppresses it
by more than 100 dB at typical bin distances. Noise-free estimates agree with the true angle to better
than 10⁻³ degrees.

### 10. Broadband AoA from GCC-PHAT (`aoa.estimate_aoa_gcc_phat`)

1. Run GCC-PHAT on **every pair** i < j, M(M−1)/2 delays in all.
2. Feed them all to one least-squares fit.

Time-delay estimation has no λ/2 constraint, so wide arrays work too. It does need broadband signal
energy, though. A pure tone's correlation is periodic, so its delay is only known modulo one period;
use the narrowband path for tones.

### 11. Least-squares angle fit

Both estimators fit the model $D_p = -b_p s / c$ over baselines $b_p = x_j - x_i$, where $s = \sin\theta$.
Minimising $\sum_p (D_p + b_p s/c)^2$ gives the closed form

$$\hat s = -c\,\frac{\sum_p D_p\, b_p}{\sum_p b_p^2}, \qquad \hat\theta = \arcsin\big(\text{clip}(\hat s, -1, 1)\big)$$

The RMS residual of that fit comes back as `residual_rms_s`, a built-in quality metric. A large residual
means multipath, more than one source, a near-field source, or a geometry error. An unclipped
$|\hat s| > 1$ is reported in `sin_theta` and flags physically inconsistent delays. Sensitivity worsens
towards endfire, since $d\theta = dD\, c / (b\cos\theta)$.

### 12. Simulation with exact fractional delays (`signals.py`)

- **Tones and multitones** are continuous functions s(t), sampled as $x_m[n] = s(n/f_s - \tau_m)$.
  This is exact, with no interpolation filter.
- **Broadband noise** is drawn in the frequency domain: independent complex Gaussian coefficients S[k]
  over the band, synthesised per channel as $x_m = \text{IRFFT}\{S[k]\, e^{-j2\pi f_k \tau_m}\}$.
  This gives an exact fractional delay of an N-periodic band-limited process with dense spectral support,
  which GCC-PHAT needs.
- **Additive white Gaussian noise** is set per channel for a target SNR:
  $\sigma^2 = P_{\text{signal}} / 10^{\text{SNR}_{\text{dB}}/10}$.

---

## Measured precision (from the test suite)

| Check | Tolerance asserted |
|---|---|
| FFT vs O(N²) DFT definition, N ∈ {1…1024, primes, 1000} | < 1e-10 · N absolute |
| `ifft(fft(x))` round trip, up to N = 4096 | < 1e-12 |
| Closed forms (on-bin cosine/sine, impulse, shifted impulse, DC, geometric series), Parseval, linearity, shift and convolution theorems | ≤ 1e-9 |
| Sub-bin tone frequency (on- and off-bin) | < 0.01 bin |
| GCC-PHAT integer delay / fractional delay | < 0.01 / < 0.05 samples |
| Narrowband AoA, noiseless, −80°…+72°, 500–1700 Hz | < 0.001° |
| Narrowband AoA at 0 dB SNR | < 1° |
| Narrowband AoA, 8-element 2.4 GHz RF array | < 0.001° |
| GCC-PHAT broadband AoA, noiseless / 0 dB SNR | < 0.15° / < 0.5° |

## Limitations

- **Far-field, single dominant source, linear array.** Two simultaneous sources at the same frequency bias
  both estimators. Subspace methods such as MUSIC or ESPRIT would separate them, and the cross-spectra
  computed here are the right input for that.
- **Left/right ambiguity.** A linear array cannot tell θ from 180° − θ (front from back).
- **Speed.** This is pure CPython. A 4096-point FFT takes a few milliseconds, which is fine for audio-rate
  block processing. For multi-MS/s SDR streams, port `_radix2` to C or decimate first.
