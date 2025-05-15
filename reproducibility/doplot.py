import numpy as np
import matplotlib.pyplot as pl
from astropy.io import fits
import torchmfbd

def crisp(save=False):    
    spot_8542 = fits.open('spot_8542/spot_8542.fits')

    nx_spot_8542 = spot_8542[0].data.shape[1]
    pix = 0.059

    qs_8542 = fits.open('qs_8542/qs_8542.fits')

    nx_qs_8542 = qs_8542[0].data.shape[1]
    pix = 0.059

    spot_sv_8542 = fits.open('spot_sv_8542/spot_sv_8542.fits')

    qs_sv_8542 = fits.open('qs_sv_8542/qs_sv_8542.fits')

    fig, ax = pl.subplots(nrows=4, ncols=4, figsize=(19, 19), sharex=True, sharey=True, tight_layout=True)

    vmins = [0.5, 0.3]
    vmaxs = [1.6, 2.1]

    loop = 0
    for i in range(4):
        for j in range(2):
            if i == 0 or i == 1:
                im = spot_8542[i].data[j, 6:-6, 6:-6]
                norm_wb = np.nanmean(spot_8542[i].data[0, 6:-6, 6:-6])
            if i == 2:
                im = spot_8542[i].data[j, 190:190+nx_spot_8542, 190:190+nx_spot_8542]                
                norm_wb = np.nanmean(spot_8542[i].data[0, ...])
            if i == 3:
                im = spot_sv_8542[1].data[j, 0, 6:-6, 6:-6]
                norm_wb = np.nanmean(spot_sv_8542[1].data[0, ...])
            
            im = im / norm_wb
                            
            ax[i, j].imshow(im, extent=[0, nx_spot_8542*pix, 0, nx_spot_8542*pix], cmap='gray', vmin=vmins[j], vmax=vmaxs[j])

            contrast = np.nanstd(im) / np.nanmean(im) * 100.0
            ax[i, j].text(0.75, 0.95, f'{contrast:.1f}%',
                          transform=ax[i, j].transAxes, 
                          fontsize=18, 
                          verticalalignment='top', 
                          color='yellow',
                          fontweight='bold')
            loop += 1

    loop = 0
    for i in range(4):
        for j in range(2):
            if i == 0 or i == 1:
                im = qs_8542[i].data[j, 6:-6, 6:-6]
                norm_wb = np.nanmean(qs_8542[i].data[0, 6:-6, 6:-6])
            if i == 2:
                im = qs_8542[i].data[j, 190:190+nx_qs_8542, 190:190+nx_qs_8542]
                norm_wb = np.nanmean(qs_8542[i].data[0, ...])
            if i == 3:                
                im = qs_sv_8542[1].data[j, 0, 6:-6, 6:-6]
                norm_wb = np.nanmean(qs_sv_8542[1].data[0, ...])
            
            im = im / norm_wb
                
            ax[i, j+2].imshow(im, extent=[0, nx_spot_8542*pix, 0, nx_spot_8542*pix], cmap='gray', vmin=vmins[j], vmax=vmaxs[j])

            contrast = np.nanstd(im) / np.nanmean(im) * 100.0
            ax[i, j+2].text(0.75, 0.95, f'{contrast:.1f}%',
                          transform=ax[i, j+2].transAxes, 
                          fontsize=18, 
                          verticalalignment='top', 
                          color='yellow',
                          fontweight='bold')
            loop += 1

    labels = ['Frame', 'torchmfbd', 'MOMFBD', 'NMF']
    for i in range(4):
        ax[i, 0].text(0.05, 0.95, labels[i], 
                      transform=ax[i, 0].transAxes, 
                      fontsize=18, 
                      verticalalignment='top', 
                      color='yellow',
                      fontweight='bold')


    fig.supxlabel('X [arcsec]')
    fig.supylabel('Y [arcsec]')

    # # Use tight_layout to adjust the layout
    # fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space at the top for colorbars

    # # Add colorbars above the first and second columns
    # cbar_ax1 = fig.add_axes([0.06, 0.97, 0.20, 0.01])  # [left, bottom, width, height]
    # cbar1 = fig.colorbar(ax[0, 0].images[0], cax=cbar_ax1, orientation='horizontal')
    # # cbar1.set_label('Intensity (Column 1)', fontsize=12)

    # cbar_ax2 = fig.add_axes([0.3, 0.97, 0.20, 0.01])  # [left, bottom, width, height]
    # cbar2 = fig.colorbar(ax[0, 1].images[0], cax=cbar_ax2, orientation='horizontal')
    # # cbar2.set_label('Intensity (Column 2)', fontsize=12)

    if save:
        pl.savefig('figs/crisp_8542.pdf', dpi=300)


def imax(save=False):
    imax = fits.open('imax/imax.fits')
    imax_pd = fits.open('imax/imaxf_image_estimated.fits')
    
    pix_imax = 0.055
    apod = 100
    nx = imax[0].data.shape[0] - 2*apod

    fig, ax = pl.subplots(nrows=2, ncols=2, figsize=(10, 10), sharex=True, sharey=True, tight_layout=True)

    for i in range(3):
        im = imax[i].data[apod:-apod, apod:-apod]
        contrast = np.nanstd(im) / np.nanmean(im) * 100.0
                
        ax.flat[i].imshow(imax[i].data[apod:-apod, apod:-apod], extent=[0, nx*pix_imax, 0, nx*pix_imax], cmap='gray')
        ax.flat[i].text(0.75, 0.95, f'{contrast:.2f}%',
                          transform=ax.flat[i].transAxes, 
                          fontsize=18, 
                          verticalalignment='top', 
                          color='yellow',
                          fontweight='bold')
    
    im = imax_pd[0].data[apod:-apod, apod:-apod]
    contrast = np.nanstd(im) / np.nanmean(im) * 100.0
    ax.flat[-1].imshow(imax_pd[0].data[apod:-apod, apod:-apod], extent=[0, nx*pix_imax, 0, nx*pix_imax], cmap='gray')
    ax.flat[-1].text(0.75, 0.95, f'{contrast:.2f}%',
                          transform=ax.flat[-1].transAxes, 
                          fontsize=18, 
                          verticalalignment='top', 
                          color='yellow',
                          fontweight='bold')

    labels = ['Focused', 'Defocused', 'torchmfbd', 'Data release']
    for i in range(4):
        ax.flat[i].text(0.05, 0.95, labels[i], 
                      transform=ax.flat[i].transAxes, 
                      fontsize=18, 
                      verticalalignment='top', 
                      color='yellow',
                      fontweight='bold')
    

    fig.supxlabel('X [arcsec]')
    fig.supylabel('Y [arcsec]')

    if save:
        pl.savefig('figs/imax.pdf', dpi=300)

def hifi(save=False):
    gband = fits.open('hifi/gband.fits')
    gband_speckle = fits.open('hifi/gband_bluec//hifiplus1_20230721_085151_sd_speckle.fts')

    tio = fits.open('hifi/tio.fits')
    tio_speckle = fits.open('hifi/ca3968_tio/hifiplus3_20230721_094744_sd_tio_speckle.fts')

    ca3968 = fits.open('hifi/ca3968.fits')
    ca3968_speckle = fits.open('hifi/ca3968_tio/hifiplus3_20230721_094744_sd_ca_speckle.fts')

    nx_gband = gband[0].data.shape[0]
    pix_gband = 0.02489

    nx_tio = tio[0].data.shape[0]
    pix_tio = 0.04979

    nx_ca3968 = ca3968[0].data.shape[0]
    pix_ca3968 = 0.02489

    vmin = 0.3
    vmax = 1.5

    fig, ax = pl.subplots(nrows=3, ncols=3, figsize=(15, 15), tight_layout=True)

    for i in range(2):
        im = gband[i].data[1:-1, 1:-1]
        im /= np.mean(im[0:120, 0:120])
        contrast = np.nanstd(im[0:120, 0:120]) / np.nanmean(im[0:120, 0:120]) * 100.0
        ax[0, i].imshow(im, extent=[0, nx_gband*pix_gband, 0, nx_gband*pix_gband], cmap='gray', vmin=vmin, vmax=vmax)
        ax[0, i].text(0.75, 0.95, f'{contrast:.2f}%',
                          transform=ax[0, i].transAxes, 
                          fontsize=18, 
                          verticalalignment='top', 
                          color='yellow',
                          fontweight='bold')
    
    im = gband_speckle[0].data[8:nx_tio-32, 8:nx_tio-32]
    im /= np.mean(im[0:120, 0:120])
    contrast = np.nanstd(im[0:120, 0:120]) / np.nanmean(im[0:120, 0:120]) * 100.0
    ax[0, 2].imshow(im, extent=[0, nx_tio*pix_tio, 0, nx_tio*pix_tio], cmap='gray', vmin=vmin, vmax=vmax)
    ax[0, 2].text(0.75, 0.95, f'{contrast:.2f}%',
                          transform=ax[0, 2].transAxes, 
                          fontsize=18, 
                          verticalalignment='top', 
                          color='yellow',
                          fontweight='bold')
    

    for i in range(2):
        im = tio[i].data[1:-1, 1:-1]
        im /= np.mean(im[0:120, 0:120])
        contrast = np.nanstd(im[0:120, 0:120]) / np.nanmean(im[0:120, 0:120]) * 100.0
        ax[1, i].imshow(im, extent=[0, nx_tio*pix_tio, 0, nx_tio*pix_tio], cmap='gray', vmin=vmin, vmax=vmax)
        ax[1, i].text(0.75, 0.95, f'{contrast:.2f}%',
                          transform=ax[1, i].transAxes, 
                          fontsize=18, 
                          verticalalignment='top', 
                          color='yellow',
                          fontweight='bold')

    im = tio_speckle[0].data[8:nx_tio-32, 8:nx_tio-32]
    im /= np.mean(im[0:120, 0:120])
    contrast = np.nanstd(im[0:120, 0:120]) / np.nanmean(im[0:120, 0:120]) * 100.0
    ax[1, 2].imshow(im, extent=[0, nx_tio*pix_tio, 0, nx_tio*pix_tio], cmap='gray', vmin=vmin, vmax=vmax)
    ax[1, 2].text(0.75, 0.95, f'{contrast:.2f}%',
                          transform=ax[1, 2].transAxes, 
                          fontsize=18, 
                          verticalalignment='top', 
                          color='yellow',
                          fontweight='bold')

    for i in range(2):
        im = ca3968[i].data[1:-1, 1:-1]
        im /= np.mean(im[-120:, -120:])
        contrast = np.nanstd(im[-120:, -120:]) / np.nanmean(im[-120:, -120:]) * 100.0
        ax[2, i].imshow(im, extent=[0, nx_ca3968*pix_ca3968, 0, nx_ca3968*pix_ca3968], cmap='gray', vmin=vmin, vmax=vmax)
        ax[2, i].text(0.75, 0.95, f'{contrast:.2f}%',
                          transform=ax[2, i].transAxes, 
                          fontsize=18, 
                          verticalalignment='top', 
                          color='yellow',
                          fontweight='bold')
        
    
    im = ca3968_speckle[0].data[8:nx_ca3968-32, 8:nx_ca3968-32]
    im /= np.mean(im[-120:, -120:])
    contrast = np.nanstd(im[-120:, -120:]) / np.nanmean(im[-120:, -120:]) * 100.0
    ax[2, 2].imshow(im, extent=[0, nx_ca3968*pix_ca3968, 0, nx_ca3968*pix_ca3968], cmap='gray', vmin=vmin, vmax=vmax)
    ax[2, 2].text(0.75, 0.95, f'{contrast:.2f}%',
                          transform=ax[2, 2].transAxes, 
                          fontsize=18, 
                          verticalalignment='top', 
                          color='yellow',
                          fontweight='bold')
    
    # Add a rectangle to indicate the region where contrast is computed
    rect = pl.Rectangle((0, (1022 - 120)*pix_gband), 120 * pix_gband, 1022 * pix_gband, linewidth=2, edgecolor='red', facecolor='none')
    ax[0, 0].add_patch(rect)
    
    rect = pl.Rectangle((0, (1022 - 120)*pix_tio), 120 * pix_tio, 1022 * pix_tio, linewidth=2, edgecolor='red', facecolor='none')
    ax[1, 0].add_patch(rect)
    
    rect = pl.Rectangle(((1022 - 120)*pix_ca3968, 0), 1022 * pix_ca3968, 120 * pix_ca3968, linewidth=2, edgecolor='red', facecolor='none')
    ax[2, 0].add_patch(rect)

    labels = ['G-band', 'TiO', 'Ca II H']

    for i in range(3):
        ax[i, 0].text(0.05, 0.95, labels[i], 
                      transform=ax[i, 0].transAxes, 
                      fontsize=18, 
                      verticalalignment='top', 
                      color='yellow',
                      fontweight='bold')


    fig.supxlabel('X [arcsec]')
    fig.supylabel('Y [arcsec]')
    ax[0, 0].set_title('Frame')
    ax[0, 1].set_title('torchmfbd')
    ax[0, 2].set_title('Speckle')

    if save:
        pl.savefig('figs/hifi.pdf', dpi=300)

def chromis(save=False):
    spot = fits.open('spot_3934/spot_3934.fits')
    spot_pd = fits.open('spot_3934/spot_3934_pd.fits')
    
    pix_spot = 0.038
    vmin = [0.1, 0.2]
    vmax = [2.1, 2.0]

    delta = 150
    
    fig, ax = pl.subplots(nrows=4, ncols=2, figsize=(10, 20), sharex=True, sharey=True, tight_layout=True)
    
    for i in range(2):
        im = spot[0].data[i, 1:-1, 1:-1]
        im /= np.mean(im[0:delta, 0:delta])

        nx = im.shape[0]

        ax[0, i].imshow(im, extent=[0, nx*pix_spot, 0, nx*pix_spot], cmap='gray', vmin=vmin[i], vmax=vmax[i])
        contrast = np.nanstd(im[0:delta, 0:delta]) / np.nanmean(im[0:delta, 0:delta]) * 100.0
        if i == 0:
            ax[0, i].text(0.75, 0.95, f'{contrast:.2f}%',
                          transform=ax[0, i].transAxes, 
                          fontsize=18, 
                          verticalalignment='top', 
                          color='yellow',
                          fontweight='bold')
        
    for i in range(2):
        im = spot[1].data[i, 1:-1, 1:-1]
        im /= np.mean(im[0:delta, 0:delta])

        ax[1, i].imshow(im, extent=[0, nx*pix_spot, 0, nx*pix_spot], cmap='gray', vmin=vmin[i], vmax=vmax[i])
        contrast = np.nanstd(im[0:delta, 0:delta]) / np.nanmean(im[0:delta, 0:delta]) * 100.0
        if i == 0:
            ax[1, i].text(0.75, 0.95, f'{contrast:.2f}%',
                          transform=ax[1, i].transAxes, 
                          fontsize=18, 
                          verticalalignment='top', 
                          color='yellow',
                          fontweight='bold')

    for i in range(2):
        im = spot_pd[1].data[i, 1:-1, 1:-1]
        im /= np.mean(im[0:delta, 0:delta])

        ax[2, i].imshow(im, extent=[0, nx*pix_spot, 0, nx*pix_spot], cmap='gray', vmin=vmin[i], vmax=vmax[i])
        contrast = np.nanstd(im[0:delta, 0:delta]) / np.nanmean(im[0:delta, 0:delta]) * 100.0
        if i == 0:
            ax[2, i].text(0.75, 0.95, f'{contrast:.2f}%',
                          transform=ax[2, i].transAxes, 
                          fontsize=18, 
                          verticalalignment='top', 
                          color='yellow',
                          fontweight='bold')
        
    for i in range(2):
        im = spot[2].data[i, 75:75+454, 535:535+454]
        im /= np.mean(im[0:delta, 0:delta])
        ax[3, i].imshow(im, extent=[0, nx*pix_spot, 0, nx*pix_spot], cmap='gray', vmin=vmin[i], vmax=vmax[i])
        contrast = np.nanstd(spot[2].data[i, 75:75+454, 535:535+454][0:delta, 0:delta]) / np.nanmean(spot[2].data[i, 75:75+454, 535:535+454][0:delta, 0:delta]) * 100.0
        if i == 0:
            ax[3, i].text(0.75, 0.95, f'{contrast:.2f}%',
                          transform=ax[3, i].transAxes, 
                          fontsize=18, 
                          verticalalignment='top', 
                          color='yellow',
                          fontweight='bold')
                
    labels = ['Frame', 'torchmfbd', 'torchmfbd PD', 'MOMFBD PD']
    for i in range(4):
        ax[i, 0].text(0.05, 0.95, labels[i], 
                      transform=ax[i, 0].transAxes, 
                      fontsize=18, 
                      verticalalignment='top', 
                      color='yellow',
                      fontweight='bold')
    
    ax[0, 0].set_title('Wideband')
    ax[0, 1].set_title('Narrowband')

    rect = pl.Rectangle((0, nx*pix_spot), delta * pix_spot, -delta * pix_spot, linewidth=2, edgecolor='red', facecolor='none')
    ax[0, 0].add_patch(rect)
    
    fig.supxlabel('X [arcsec]')
    fig.supylabel('Y [arcsec]')

    if save:
        pl.savefig('figs/spot_3934.pdf', dpi=300)

def timing(save=False):

    patches = np.array([961, 225, 49])

    tmp_32 = np.load('timing/times_32.npz')
    tmp_64 = np.load('timing/times_64.npz')
    tmp_128 = np.load('timing/times_128.npz')
    # tmp_256 = np.load('timing/times_256.npz')

    fig, ax = pl.subplots(figsize=(8, 6), tight_layout=True)
    ax.tick_params(axis='both', which='major', labelsize=14)  # Increase tick label size
    ax.set_title(ax.get_title(), fontsize=16)  # Increase title font size
    ax.set_xlabel(ax.get_xlabel(), fontsize=14)  # Increase x-axis label font size
    ax.set_ylabel(ax.get_ylabel(), fontsize=14)  # Increase y-axis label font size

    x_32 = np.clip(patches[0] / tmp_32['values'], a_min=1, a_max=None)
    x_64 = np.clip(patches[1] / tmp_64['values'], a_min=1, a_max=None)
    x_128 = np.clip(patches[2] / tmp_128['values'], a_min=1, a_max=None)
            
    ax.plot(x_32, tmp_32['times'], linewidth=3, label=f'32 pix')
    ax.plot(x_64, tmp_64['times'], linewidth=3, label=f'64 pix')
    ax.plot(x_128, tmp_128['times'], linewidth=3, label=f'128 pix')
    
    ax2 = ax.twinx()
    ax2.plot(x_32, tmp_32['mem'] / 1024, '--', linewidth=3)
    ax2.plot(x_64, tmp_64['mem'] / 1024, '--', linewidth=3)
    ax2.plot(x_128, tmp_128['mem'] / 1024, '--', linewidth=3)

    ax.text(0.75, 0.18, 'Memory',
                      transform=ax.transAxes, 
                      fontsize=14, 
                      verticalalignment='top')
    
    ax.text(0.75, 0.55, 'Time',
                      transform=ax.transAxes, 
                      fontsize=14, 
                      verticalalignment='top')
    
    ax.set_title(r'512$\times$512')
    
    ax.set_xlim([0, 12])
    ax.set_ylim([1., 5])
    ax.legend(loc='upper right', fontsize=14)
    ax.set_xlabel('Number of batches')
    ax.set_ylabel('Time [s]')
    ax2.set_ylabel('Memory [GB]')

    if save:
        pl.savefig('figs/timing.pdf', dpi=300)


def power(save=False):

    fig, ax = pl.subplots(nrows=2, ncols=3, figsize=(18, 10), tight_layout=True)

    #*******************
    # HIFI
    #*******************
    gband = fits.open('hifi/gband.fits')
    gband_speckle = fits.open('hifi/gband_bluec//hifiplus1_20230721_085151_sd_speckle.fts')

    tio = fits.open('hifi/tio.fits')
    tio_speckle = fits.open('hifi/ca3968_tio/hifiplus3_20230721_094744_sd_tio_speckle.fts')

    ca3968 = fits.open('hifi/ca3968.fits')
    ca3968_speckle = fits.open('hifi/ca3968_tio/hifiplus3_20230721_094744_sd_ca_speckle.fts')

    nx_gband = gband[0].data.shape[0]
    pix_gband = 0.02489
    diff_gband = 1.22 * 4300e-8 / 144.0 * 206265.0

    nx_tio = tio[0].data.shape[0]
    pix_tio = 0.04979
    diff_tio = 1.22 * 7058e-8 / 144.0 * 206265.0

    nx_ca3968 = ca3968[0].data.shape[0]
    pix_ca3968 = 0.02489
    diff_ca3968 = 1.22 * 3968e-8 / 144.0 * 206265.0

    nx_tio = tio[0].data.shape[0]
    

    # G-band
    kk, power = torchmfbd.util.azimuthal_power(gband[0].data[1:-1, 1:-1], apodization=10, angles=[-45,45], range_angles=15)    
    ax[0, 0].loglog(kk, power / power[0], label='Frame', linewidth=2)
    upper = np.nanmean(power[0:10] / power[0])
    lower = np.nanmean(power[-10:] / power[0])

    kk, power = torchmfbd.util.azimuthal_power(gband[1].data[1:-1, 1:-1], apodization=10, angles=[-45,45], range_angles=15)
    ax[0, 0].loglog(kk, power / power[0], label='torchmfbd', linewidth=2)

    kk, power = torchmfbd.util.azimuthal_power(gband_speckle[0].data[8:nx_tio-32, 8:nx_tio-32][1:-1, 1:-1], apodization=10, angles=[-45,45], range_angles=15)
    ax[0, 0].loglog(kk, power / power[0], label='Speckle', linewidth=2)
    ax[0, 0].axvline(1.0 / (diff_gband / pix_gband), color='black')
    ax[0, 0].legend()
    ax[0, 0].set_title('HiFI - G-band')
    ax[0, 0].set_ylim([lower / 100., 10.0*upper])
    ax[0, 0].set_xlim([3e-3, 0.6])

    # TiO
    kk, power = torchmfbd.util.azimuthal_power(tio[0].data[1:-1, 1:-1], apodization=10, angles=[-45,45], range_angles=15)
    ax[0, 1].loglog(kk, power / power[0], label='Frame', linewidth=2)
    upper = np.nanmean(power[0:10] / power[0])
    lower = np.nanmean(power[-10:] / power[0])

    kk, power = torchmfbd.util.azimuthal_power(tio[1].data[1:-1, 1:-1], apodization=10, angles=[-45,45], range_angles=15)
    ax[0, 1].loglog(kk, power / power[0], label='torchmfbd', linewidth=2)

    kk, power = torchmfbd.util.azimuthal_power(tio_speckle[0].data[8:nx_tio-32, 8:nx_tio-32][1:-1, 1:-1], apodization=10, angles=[-45,45], range_angles=15)
    ax[0, 1].loglog(kk, power / power[0], label='Speckle', linewidth=2)
    ax[0, 1].axvline(1.0 / (diff_tio / pix_tio), color='black')
    ax[0, 1].legend()
    ax[0, 1].set_title('HiFI - TiO')
    ax[0, 1].set_ylim([lower / 100., 10.0*upper])
    ax[0, 1].set_xlim([3e-3, 0.6])

    # Ca II H
    kk, power = torchmfbd.util.azimuthal_power(ca3968[0].data[1:-1, 1:-1], apodization=10, angles=[-45,45], range_angles=15)
    ax[0, 2].loglog(kk, power / power[0], label='Frame', linewidth=2)
    upper = np.nanmean(power[0:10] / power[0])
    lower = np.nanmean(power[-10:] / power[0])

    kk, power = torchmfbd.util.azimuthal_power(ca3968[1].data[1:-1, 1:-1], apodization=10, angles=[-45,45], range_angles=15)
    ax[0, 2].loglog(kk, power / power[0], label='torchmfbd', linewidth=2)

    kk, power = torchmfbd.util.azimuthal_power(ca3968_speckle[0].data[8:nx_ca3968-32, 8:nx_ca3968-32][1:-1, 1:-1], apodization=10, angles=[-45,45], range_angles=15)
    ax[0, 2].loglog(kk, power / power[0], label='Speckle', linewidth=2)
    ax[0, 2].axvline(1.0 / (diff_ca3968 / pix_ca3968), color='black')
    ax[0, 2].legend()
    ax[0, 2].set_title('HiFI - Ca II H')
    ax[0, 2].set_ylim([lower / 100., 10.0*upper])
    ax[0, 2].set_xlim([3e-3, 0.6])

    #*******************
    # CRISP
    #*******************
    qs_8542 = fits.open('qs_8542/qs_8542.fits')
    # mfbd = np.copy(readsav('aux/patches_momfbd_qs8542_camXX_00010_+65_wb.sav')['patches'])
    
    nx_qs_8542 = qs_8542[0].data.shape[1]
    diff_crisp = 1.22 * 8542e-8 / 100.0 * 206265.0
    pix_crisp = 0.059

    # kk, power = torchmfbd.util.azimuthal_power(qs_8542[5].data[20, 0, ...])
    # ax[1, 0].loglog(kk, power / power[0], label='Frame', linewidth=2)
        
    # kk, power = torchmfbd.util.azimuthal_power(qs_8542[3].data[20, ...])
    # ax[1, 0].loglog(kk, power / power[0], label='torchmfbd', linewidth=2)

    # kk, power = torchmfbd.util.azimuthal_power(mfbd[1, 1, :, :])
    # ax[1, 0].loglog(kk, power / power[0], label='MOMFBD', linewidth=2)
    
    kk, power = torchmfbd.util.azimuthal_power(qs_8542[0].data[0, ...][1:-1, 1:-1], apodization=10, angles=[-45,45], range_angles=15)
    ax[1, 0].loglog(kk, power / power[0], label='Frame', linewidth=2)    
    upper = np.nanmean(power[0:10] / power[0])
    lower = np.nanmean(power[-10:] / power[0])

    kk, power = torchmfbd.util.azimuthal_power(qs_8542[1].data[0, ...][1:-1, 1:-1], apodization=10, angles=[-45,45], range_angles=15)
    ax[1, 0].loglog(kk, power / power[0], label='torchmfbd', linewidth=2)
    kk, power = torchmfbd.util.azimuthal_power(qs_8542[2].data[0, 190:190+nx_qs_8542, 190:190+nx_qs_8542], apodization=10, angles=[-45,45], range_angles=15)
    ax[1, 0].loglog(kk, power / power[0], label='MOMFBD', linewidth=2)    

    ax[1, 0].legend()
    ax[1, 0].axvline(1.0 / (diff_crisp / pix_crisp), color='black')
    ax[1, 0].set_title('CRISP - QS WB')
    ax[1, 0].set_ylim([lower / 10.0, 10.0*upper])
    ax[1, 0].set_xlim([3e-3, 0.6])

    
    #*******************
    # CHROMIS
    #*******************
    chromis = fits.open('spot_3934/spot_3934.fits')
    chromis_pd = fits.open('spot_3934/spot_3934_pd.fits')

    nx = chromis[1].data.shape[1]
    pix_chromis = 0.038

    diff_chromis = 1.22 * 3934e-8 / 100.0 * 206265.0
    pix_chromis = 0.038

    kk, power = torchmfbd.util.azimuthal_power(chromis[0].data[0, ...], apodization=10, angles=[-45,45], range_angles=15)
    ax[1, 1].loglog(kk, power / power[0], label='Frame', linewidth=2)
    upper = np.nanmean(power[0:10] / power[0])
    lower = np.nanmean(power[-10:] / power[0])

    kk, power = torchmfbd.util.azimuthal_power(chromis[1].data[0, ...][1:-1,1:-1], apodization=10, angles=[-45,45], range_angles=15)
    ax[1, 1].loglog(kk, power / power[0], label='torchmfbd', linewidth=2)    
    kk, power = torchmfbd.util.azimuthal_power(chromis_pd[1].data[0, ...], apodization=10, angles=[-45,45], range_angles=15)
    ax[1, 1].loglog(kk, power / power[0], label='torchmfbd PD', linewidth=2)
    kk, power = torchmfbd.util.azimuthal_power(chromis[2].data[0, 72:72+nx, 525:525+nx], apodization=10, angles=[-45,45], range_angles=15)
    ax[1, 1].loglog(kk, power / power[0], label='MOMFBD PD', linewidth=2)
    ax[1, 1].legend()
    ax[1, 1].axvline(1.0 / (diff_chromis / pix_chromis), color='black')
    ax[1, 1].set_title('CHROMIS - WB')
    ax[1, 1].set_ylim([lower / 10.0, 10.0*upper])
    ax[1, 1].set_xlim([3e-3, 0.6])

    fig.supxlabel(r'Spatial frequency [pix$^{-1}$]')
    fig.supylabel('Normalized azimuthally averaged power')

    #*******************
    # IMaX
    #*******************
    imax = fits.open('imax/imax.fits')
    imax_pd = fits.open('imax/imaxf_image_estimated.fits')

    apod = 100
    nx_qs_8542 = imax[0].data.shape[0] - 2*apod
    pix_imax = 0.055

    diff_imax = 1.22 * 5250e-8 / 100.0 * 206265.0
    pix_imax = 0.055

    kk, power = torchmfbd.util.azimuthal_power(imax[0].data[apod:-apod, apod:-apod], apodization=10, angles=[-45,45], range_angles=15)
    ax[1, 2].loglog(kk, power / power[0], label='Frame', linewidth=2)
    upper = np.nanmean(power[0:10] / power[0])
    lower = np.nanmean(power[-10:] / power[0])

    kk, power = torchmfbd.util.azimuthal_power(imax[2].data[apod:-apod, apod:-apod], apodization=10, angles=[-45,45], range_angles=15)
    ax[1, 2].loglog(kk, power / power[0], label='torchmfbd', linewidth=2)
    kk, power = torchmfbd.util.azimuthal_power(imax_pd[0].data[apod:-apod, apod:-apod], apodization=10, angles=[-45,45], range_angles=15)
    ax[1, 2].loglog(kk, power / power[0], label='Data release', linewidth=2)
    ax[1, 2].legend()
    ax[1, 2].axvline(1.0 / (diff_imax / pix_imax), color='black')
    ax[1, 2].set_title('IMaX')
    ax[1, 2].set_ylim([lower / 2.0, 10.0*upper])
    ax[1, 2].set_xlim([3e-3, 0.6])
    

    if save:
        pl.savefig('figs/power.pdf', dpi=300)



if __name__ == '__main__':

    pl.close('all')

    save = False

    crisp(save)

    imax(save)
        
    hifi(save)

    chromis(save)
        
    timing(save)

    power(save)
