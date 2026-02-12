import numpy as np
import matplotlib.pyplot as pl

def log_evidence2(p_hat, I_hat, K, f0, p, sigma):

    s_u = K / (1 + (f / f0)**p)
    
    term1 = np.abs(I_hat)**2 / sigma**2

    Q = sigma**2 + s_u * np.abs(p_hat)**2
    term2 = s_u * np.abs(np.conj(p_hat) * I_hat)**2 / (sigma**2 * Q)
    
    term3 = np.log(Q)

    # print(term2)
    
    return 0.5 * np.sum(term1 - term2 + term3) + np.log(K) + np.log(f0)

def log_evidence(p_hat, I_hat, K, f0, p, sigma):

    s_u = K / (1 + (f / f0)**p)

    du2 = I_hat * np.conj(I_hat)
    hu2 = p_hat * np.conj(p_hat)
    hu_du = I_hat * np.conj(p_hat)
    hu_du2 = hu_du * np.conj(hu_du)

    Q = sigma**2 + s_u * hu2

    log_like = 0.5 * (du2 - s_u * hu_du2 / Q) / sigma**2
    log_prior = 0.5  * np.log(Q)

    # print(s_u * hu_du2 / (Q*sigma**2))

    return np.sum(log_like + log_prior) + np.log(K) + np.log(f0)

N = 128
freq = np.fft.fftfreq(N)  # cycles / pixel
f = np.abs(freq)

K_true = 10.0
f0_true = 0.8
p_true = 2.0

s_u = K_true / (1 + (f / f0_true)**p_true)

rng = np.random.default_rng(0)

o_hat = (
    np.sqrt(s_u / 2)
    * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
)

x = np.arange(N)
psf = np.exp(-0.5 * ((x - N//2) / 2)**2)
psf /= psf.sum()  # unit area
p_hat = np.fft.fft(psf, norm="ortho")

sigma = 0.05

noise = (
    sigma / np.sqrt(2)
    * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
)

I_hat = p_hat * o_hat + noise

logKs = np.linspace(-4, 4, 50)
logf0s = np.linspace(-3, 3, 50)

L = np.zeros((len(logKs), len(logf0s)))
L2 = np.zeros((len(logKs), len(logf0s)))

for i, logK in enumerate(logKs):
    for j, logf0 in enumerate(logf0s):
        K = 10**logK
        f0 = 10**logf0
        L[i, j] = log_evidence(p_hat, I_hat, K, f0, p_true, sigma)
        L2[i, j] = log_evidence2(p_hat, I_hat, K, f0, p_true, sigma)

        # breakpoint()

fig, ax = pl.subplots(nrows=1, ncols=2, figsize=(12, 5))
ax[0].imshow(L.T, origin="lower", extent=(logKs[0], logKs[-1], logf0s[0], logf0s[-1]), aspect="auto")
ax[0].plot(np.log10(K_true), np.log10(f0_true), "x", color="red")

ax[1].imshow(L2.T, origin="lower", extent=(logKs[0], logKs[-1], logf0s[0], logf0s[-1]), aspect="auto")
ax[1].plot(np.log10(K_true), np.log10(f0_true), "x", color="red")

peak_idx = np.unravel_index(np.argmin(L), L.shape)
ax[0].plot(logKs[peak_idx[0]], logf0s[peak_idx[1]], "o", color="white", markersize=8)

peak_idx = np.unravel_index(np.argmin(L2), L2.shape)
ax[1].plot(logKs[peak_idx[0]], logf0s[peak_idx[1]], "o", color="white", markersize=8)
ax[0].set_xlabel("log10(K)")
ax[0].set_ylabel("log10(f0)")
ax[1].set_xlabel("log10(K)")
ax[1].set_ylabel("log10(f0)")
pl.show()