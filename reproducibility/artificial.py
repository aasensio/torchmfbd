import numpy as np
import matplotlib.pyplot as pl
import torchmfbd

def generate_image_with_aapsd(size, D, wavelength, pixel_size, K, v0, beta, mean_image):
    """
    Generates an image with a power spectral density proportional to 1/f^beta.
    size: tuple (width, height)
    beta: exponent (0 = white noise, 1 = pink noise, 2 = brown noise)
    """

    cutoff = D / (wavelength * 1e-8) / 206265.0


    w, h = size
    # 1. Create White Noise
    white_noise = np.random.standard_normal(size)
    
    # 2. Fourier Transform
    noise_fft = np.fft.fft2(white_noise, norm='ortho')
    
    # 3. Create a radial frequency grid
    # Get coordinates relative to the center
    u = np.fft.fftfreq(w, d=pixel_size) / cutoff
    v = np.fft.fftfreq(h, d=pixel_size) / cutoff
    u, v = np.meshgrid(u, v)
    
    # Calculate radial frequency f = sqrt(u^2 + v^2)
    rho = np.sqrt(u**2 + v**2)    
    
    # 4. Generate the filter (Square root because PSD is power, we need amplitude)
    # Power Spectral Density P(f) ~ 1 / f^beta
    # Amplitude A(f) ~ sqrt(P(f)) ~ 1 / f^(beta/2)
    psd = K / (1.0 + (rho/v0)**2)**(beta / 2.0)
    
    f_filter = np.sqrt(psd)
    
    # 5. Apply filter and Inverse FFT
    filtered_fft = noise_fft * f_filter
    filtered_fft[rho == 0.0] = np.sqrt(w*h) * mean_image # Set DC component to 0

    img_back = np.fft.ifft2(filtered_fft, norm='ortho')
    
    # Return real part and normalize
    result = np.real(img_back)
    return result, f_filter

pl.close('all')
# Example: Generate 'Pink Noise' (beta=1)

KK = np.zeros(9)
mn = np.zeros(9)
for i, Ks in enumerate([1.0, 3, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]):
    D = 100.0
    wavelength = 8542
    pixel_size = 0.059
    K = Ks
    v0 = 0.1
    beta = 2.0
    mean_image = 1.0
    noise = 0.5
    cutoff = D / (wavelength * 1e-8) / 206265.0

    img, psd = generate_image_with_aapsd((512, 512), D=D, wavelength=wavelength, pixel_size=pixel_size, K=K, v0=v0, beta=beta, mean_image=mean_image)

    fig, ax = pl.subplots(nrows=1, ncols=2, figsize=(12, 6))

    tmp = ax[0].imshow(img, cmap='gray')
    pl.colorbar(tmp, ax=ax[0])
    ax[0].set_title("Synthesized Image (AAPSD $1/f^2$)")
    ax[0].axis('off')

    im_noise = img + noise * np.random.normal(size=img.shape)
    f = np.fft.fft2(im_noise, norm='ortho')
    power2d = np.abs(f)**2 / (img.shape[0] * img.shape[1])

    kk, power = torchmfbd.azimuthal_power(im_noise)

    nu = kk / cutoff / pixel_size

    ax[1].loglog(kk, power)
    ax[1].loglog(kk, K / (1.0 + (nu/v0)**2)**(beta / 2.0))
    ax[1].axhline(noise**2, color='r', linestyle='--', label='Cutoff Frequency')

    print(f"Mean value of the image: {np.mean(img)} - noisy: {np.mean(im_noise)} - {np.sqrt(power2d[0, 0])}")
    print(f"Std value of the image: {np.std(img)} - noisy: {np.std(im_noise)}")

    KK[i] = K
    mn[i] = np.std(im_noise)