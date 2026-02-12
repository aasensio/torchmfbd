import numpy as np
import matplotlib.pyplot as pl

pl.close('all')

sx = 1.0
sy = 0.2
rxy = 0.1

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

pix_size = x[1] - x[0]

xy = (X/sx)**2 + (Y/sy)**2 - 2.0*rxy*X*Y/(sx*sy)
psf = np.exp(-0.5 * xy / (1.0 - rxy**2))
psf = psf / (2.0 * np.pi * sx * sy * np.sqrt(1.0 - rxy**2))

psf = np.fft.fftshift(psf)

psf_f = np.fft.fft2(psf, norm='ortho')

f_x = np.fft.fftfreq(psf.shape[0], d=(x[1] - x[0]))
f_y = np.fft.fftfreq(psf.shape[1], d=(y[1] - y[0]))
F_X, F_Y = np.meshgrid(f_x, f_y)

otf = np.exp(-2.0 * np.pi**2 * (F_X**2 * sx**2 + F_Y**2 * sy**2 + 2.0*rxy*F_X*F_Y*sx*sy))

fig, ax = pl.subplots(nrows=2, ncols=3, figsize=(15, 10))

im = ax[0, 0].imshow(psf_f.real)
pl.colorbar(im, ax=ax[0, 0])
ax[0, 0].set_title('PSF Fourier Transform (Real Part)')

im = ax[0, 1].imshow(np.fft.fftshift(np.fft.ifft2(psf_f, norm='ortho').real), extent=(x[0], x[-1], y[0], y[-1]))
pl.colorbar(im, ax=ax[0, 1])
ax[0, 1].set_title('Inverse FFT of PSF Fourier Transform (Real Part)')

im = ax[0, 2].imshow(np.fft.fftshift(psf), extent=(x[0], x[-1], y[0], y[-1]))
pl.colorbar(im, ax=ax[0, 2])

im = ax[1, 0].imshow(otf)
pl.colorbar(im, ax=ax[1, 0])
ax[1, 0].set_title('OTF')

im = ax[1, 1].imshow(np.fft.fftshift(np.fft.ifft2(otf, norm='ortho').real), extent=(x[0], x[-1], y[0], y[-1]))
pl.colorbar(im, ax=ax[1, 1])
ax[1, 1].set_title('Inverse FFT of OTF (Real Part)')

area_psf = np.sum(psf) * pix_size**2
area_psf_recovered = np.sum(np.fft.ifft2(psf_f, norm='ortho').real) * pix_size**2

area_psf2 = np.sum(np.fft.ifft2(otf, norm='ortho').real) * pix_size**2

print(f"Area of PSF original: {area_psf}")
print(f"Area of PSF recovered from Inverse Fourier Transform: {area_psf_recovered}")
print(f"Area of PSF recovered from OTF: {area_psf2}")