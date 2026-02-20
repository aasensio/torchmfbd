import numpy as np
import matplotlib.pyplot as pl
from astropy.io import fits
import torchmfbd

def crisp(save=False):
    spot_8542_joint = fits.open('spot_8542/spot_8542_joint.fits')
    spot_8542_marginal = fits.open('spot_8542/spot_8542_marginal.fits')

    nx_spot_8542 = spot_8542_joint[0].data.shape[1]
    pix = 0.059

    qs_8542_joint = fits.open('qs_8542/qs_8542_joint.fits')
    qs_8542_marginal = fits.open('qs_8542/qs_8542_marginal.fits')

    nx_qs_8542 = qs_8542_joint[0].data.shape[1]
    pix = 0.059

    fig, ax = pl.subplots(nrows=4, ncols=4, figsize=(19, 19), sharex=True, sharey=True, tight_layout=True)

    vmins = [0.5, 0.3]
    vmaxs = [1.6, 2.1]

    loop = 0
    for i in range(4):
        for j in range(2):
            if i == 0:                
                im = spot_8542_joint[0].data[j, 6:-6, 6:-6]
                norm_wb = np.nanmean(im)
            if i == 1:
                im = spot_8542_joint[1].data[j, 6:-6, 6:-6]
                norm_wb = np.nanmean(im)
            if i == 2:
                im = spot_8542_marginal[1].data[j, 6:-6, 6:-6]
                norm_wb = np.nanmean(im)
            if i == 3:
                im = spot_8542_joint[2].data[j, 190:190+nx_spot_8542, 190:190+nx_spot_8542]
                norm_wb = np.nanmean(im)
            
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
            if i == 0:
                im = qs_8542_joint[0].data[j, 6:-6, 6:-6]
                norm_wb = np.nanmean(im)
            if i == 1:
                im = qs_8542_joint[1].data[j, 6:-6, 6:-6]
                norm_wb = np.nanmean(im)
            if i == 2:
                im = qs_8542_marginal[1].data[j, 6:-6, 6:-6]
                norm_wb = np.nanmean(im)
            if i == 3:                
                im = qs_8542_marginal[2].data[j, 190:190+nx_qs_8542, 190:190+nx_qs_8542]
                norm_wb = np.nanmean(im)
            
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

    labels = ['Frame', 'Joint', 'Marginal', 'MOMFBD']
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
        pl.savefig('figs_marginal/images_8542.pdf', dpi=300)


    fig, ax = pl.subplots(nrows=2, ncols=1, figsize=(7.5, 15), tight_layout=True, sharex=True)
    diff_crisp = 1.22 * 8542e-8 / 100.0 * 206265.0
    pix_crisp = 0.059

    # QS    
    im = qs_8542_joint[0].data[0, ...][1:-1, 1:-1]
    kk, power = torchmfbd.util.azimuthal_power(im / np.nanmean(im), apodization=10, angles=[-45,45], range_angles=15)
    pars_s0 = np.mean(qs_8542_marginal[5].data, axis=0)
    K, v0, p, s2 = pars_s0
    cutoff = 100.0 / (8542 * 1e-8) / 206265.0
    nu = kk / cutoff / pix_crisp
    # s_u = K / (1.0 + (nu / v0)**p) / im.shape[0]
    s_u = K / (1.0 + (nu/v0)**2)**p

    s_u /= im.shape[0]

    ax[0].loglog(kk, s_u , label=r'S$_u$', linewidth=2, linestyle='--', color='C0')
    # ax[, 1].loglog(nu, s_u , label=r'S$_u$', linewidth=2, linestyle='--', color='C0')

    
    print(f'QS - K={K}, v0={v0}, p={p}')


    ax[0].loglog(kk, power / 10.0**np.nanmean(np.log10(power[5:8])) , label='Frame', linewidth=2, color='C1')
    
    im = qs_8542_joint[1].data[0, ...][1:-1, 1:-1]    
    kk, power = torchmfbd.util.azimuthal_power(im / np.nanmean(im), apodization=10, angles=[-45,45], range_angles=15)
    nu = kk / cutoff / pix_crisp
    
    ax[0].loglog(kk, power / 10.0**np.nanmean(np.log10(power[5:8])) , label='Joint', linewidth=2, color='C2')
    
    im = qs_8542_marginal[1].data[0, ...][1:-1, 1:-1]
    kk, power = torchmfbd.util.azimuthal_power(im / np.nanmean(im), apodization=10, angles=[-45,45], range_angles=15)
    nu = kk / cutoff / pix_crisp
    ax[0].loglog(kk, power / 10.0**np.nanmean(np.log10(power[5:8])) , label='Marginal', linewidth=2, color='C3')
    
    im = qs_8542_joint[2].data[0, 190:190+nx_qs_8542, 190:190+nx_qs_8542]    
    kk, power = torchmfbd.util.azimuthal_power(im / np.nanmean(im), apodization=10, angles=[-45,45], range_angles=15)
    nu = kk / cutoff / pix_crisp
    ax[0].loglog(kk, power / 10.0**np.nanmean(np.log10(power[5:8])) , label='MOMFBD', linewidth=2, color='C4')
    
    
    
    ax[0].set_ylim([1e-8, 5e1])
    ax[0].set_xlim([3e-3, 0.6])
    # ax[0 1].set_ylim([1e-8, 5e1])
    # ax[0, 1].set_xlim([1e-2, 5])

    
    # Spot    
    im = spot_8542_joint[0].data[0, ...][1:-1, 1:-1]
    kk, power = torchmfbd.util.azimuthal_power(im / np.nanmean(im), apodization=10, angles=[-45,45], range_angles=15)    
    pars_s0 = np.mean(spot_8542_marginal[5].data, axis=0)
    K, v0, p, s2 = pars_s0
    cutoff = 100.0 / (8542 * 1e-8) / 206265.0
    nu = kk / cutoff / pix_crisp
    # s_u = K / (1.0 + (nu / v0)**p) / im.shape[0]
    s_u = K / (1.0 + (nu/v0)**2)**p
    s_u /= im.shape[0]
    print(f'Spot - K={K}, v0={v0}, p={p}')

    ax[1].loglog(kk, s_u , label=r'S$_u$', linewidth=2, linestyle='--', color='C0')

    ax[1].loglog(kk, power / 10.0**np.nanmean(np.log10(power[5:8])) , label='Frame', linewidth=2, color='C1')
    upper = np.nanmean(power[0:10] )
    lower = np.nanmean(power[-10:] )

    im = spot_8542_joint[1].data[0, ...][1:-1, 1:-1]
    kk, power = torchmfbd.util.azimuthal_power(im / np.nanmean(im), apodization=10, angles=[-45,45], range_angles=15)
    nu = kk / cutoff / pix_crisp

    ax[1].loglog(kk, power / 10.0**np.nanmean(np.log10(power[5:8])) , label='Joint', linewidth=2, color='C2')
    
    im = spot_8542_marginal[1].data[0, ...][1:-1, 1:-1]
    kk, power = torchmfbd.util.azimuthal_power(im / np.nanmean(im), apodization=10, angles=[-45,45], range_angles=15)
    nu = kk / cutoff / pix_crisp

    ax[1].loglog(kk, power / 10.0**np.nanmean(np.log10(power[5:8])) , label='Marginal', linewidth=2, color='C3')
    
    im = spot_8542_joint[2].data[0, 190:190+nx_qs_8542, 190:190+nx_qs_8542]
    kk, power = torchmfbd.util.azimuthal_power(im / np.nanmean(im), apodization=10, angles=[-45,45], range_angles=15)
    nu = kk / cutoff / pix_crisp

    ax[1].loglog(kk, power / 10.0**np.nanmean(np.log10(power[5:8])) , label='MOMFBD', linewidth=2, color='C4')    
    

    ax[0].legend()
    fig.supxlabel('Spatial frequency [1/pix]')
    fig.supylabel('Normalized power')
    
    ax[0].axvline(1.0 / (diff_crisp / pix_crisp), color='black')    
    ax[1].axvline(1.0 / (diff_crisp / pix_crisp), color='black')    
    ax[0].set_title('CRISP - QS WB')
    
    ax[1].set_title('CRISP - Spot WB')
    
    
    ax[1].set_ylim([1e-8, 5e1])
    ax[1].set_xlim([3e-3, 0.6])

    if save:
        pl.savefig('figs_marginal/power_8542.pdf', dpi=300)
    

def hifi(save=False):
    gband_joint = fits.open('hifi/gband_joint.fits')
    gband_marginal = fits.open('hifi/gband_marginal.fits')
    gband_speckle = fits.open('obs/gband_bluec/hifiplus1_20230721_085151_sd_speckle.fts')

    tio = fits.open('hifi/tio.fits')
    tio_speckle = fits.open('obs/ca3968_tio/hifiplus3_20230721_094744_sd_tio_speckle.fts')

    ca3968 = fits.open('hifi/ca3968.fits')
    ca3968_speckle = fits.open('obs/ca3968_tio/hifiplus3_20230721_094744_sd_ca_speckle.fts')

    nx_gband = gband_joint[0].data.shape[0]
    pix_gband = 0.02489

    nx_tio = tio[0].data.shape[0]
    pix_tio = 0.04979

    nx_ca3968 = ca3968[0].data.shape[0]
    pix_ca3968 = 0.02489

    vmin = 0.3
    vmax = 1.5

    fig, ax = pl.subplots(nrows=1, ncols=4, figsize=(20, 5), tight_layout=True)

    for i in range(2):
        im = gband_joint[i].data[1:-1, 1:-1]
        im /= np.mean(im[0:120, 0:120])
        contrast = np.nanstd(im[0:120, 0:120]) / np.nanmean(im[0:120, 0:120]) * 100.0
        ax[i].imshow(im, extent=[0, nx_gband*pix_gband, 0, nx_gband*pix_gband], cmap='gray', vmin=vmin, vmax=vmax)
        ax[i].text(0.7, 0.95, f'{contrast:.2f}%',
                          transform=ax[i].transAxes, 
                          fontsize=18, 
                          verticalalignment='top', 
                          color='yellow',
                          fontweight='bold')
        
    im = gband_marginal[1].data[1:-1, 1:-1]
    im /= np.mean(im[0:120, 0:120])
    contrast = np.nanstd(im[0:120, 0:120]) / np.nanmean(im[0:120, 0:120]) * 100.0
    ax[2].imshow(im, extent=[0, nx_gband*pix_gband, 0, nx_gband*pix_gband], cmap='gray', vmin=vmin, vmax=vmax)
    ax[2].text(0.7, 0.95, f'{contrast:.2f}%',
                        transform=ax[2].transAxes, 
                        fontsize=18, 
                        verticalalignment='top', 
                        color='yellow',
                        fontweight='bold')
    
    im = gband_speckle[0].data[8:nx_tio-32, 8:nx_tio-32]
    im /= np.mean(im[0:120, 0:120])
    contrast = np.nanstd(im[0:120, 0:120]) / np.nanmean(im[0:120, 0:120]) * 100.0
    ax[3].imshow(im, extent=[0, nx_tio*pix_tio, 0, nx_tio*pix_tio], cmap='gray', vmin=vmin, vmax=vmax)
    ax[3].text(0.7, 0.95, f'{contrast:.2f}%',
                          transform=ax[3].transAxes, 
                          fontsize=18, 
                          verticalalignment='top', 
                          color='yellow',
                          fontweight='bold')
    
    # Add a rectangle to indicate the region where contrast is computed
    rect = pl.Rectangle((0, (1022 - 120)*pix_gband), 120 * pix_gband, 1022 * pix_gband, linewidth=2, edgecolor='red', facecolor='none')
    ax[0].add_patch(rect)

    ax[0].text(0.15, 0.95, 'G-band', 
                      transform=ax[0].transAxes, 
                      fontsize=18, 
                      verticalalignment='top', 
                      color='yellow',
                      fontweight='bold')

            
    fig.supxlabel('X [arcsec]')
    fig.supylabel('Y [arcsec]')
    ax[0].set_title('Frame')
    ax[1].set_title('Joint')
    ax[2].set_title('Marginal')
    ax[3].set_title('Speckle')

    if save:
        pl.savefig('figs_marginal/images_hifi.pdf', dpi=300)

    fig, ax = pl.subplots(nrows=1, ncols=1, figsize=(5, 5), tight_layout=True)
    diff_hifi = 1.22 * 4300e-8 / 144.0 * 206265.0
    pix_hifi = 0.02761

    # QS    
    im = gband_joint[0].data[1:-1, 1:-1]
    kk, power = torchmfbd.util.azimuthal_power(im / np.nanmean(im), apodization=10, angles=[-45,45], range_angles=15)
    
    pars_s0 = np.mean(gband_marginal[2].data, axis=0)
    K, v0, p, s2 = pars_s0
    cutoff = 100.0 / (8542 * 1e-8) / 206265.0
    nu = kk / cutoff / pix_hifi
    s_u = K / (1.0 + (nu/v0)**2)**p

    s_u /= im.shape[0]

    
    print(f'QS - K={K}, v0={v0}, p={p}')


    ax.loglog(kk, power / 10.0**np.nanmean(np.log10(power[0:5])), label='Frame', linewidth=2)    
    
    ax.loglog(kk, s_u / s_u[0], label=r'S$_u$', linewidth=2, linestyle='--')
    
    im = gband_joint[1].data[1:-1, 1:-1]
    kk, power = torchmfbd.util.azimuthal_power(im / np.nanmean(im), apodization=10, angles=[-45,45], range_angles=15)
    nu = kk / cutoff / pix_hifi
    
    ax.loglog(kk, power / 10.0**np.nanmean(np.log10(power[0:5])) , label='Joint', linewidth=2)
    
    im = gband_speckle[0].data[8:nx_tio-32, 8:nx_tio-32]
    kk, power = torchmfbd.util.azimuthal_power(im / np.nanmean(im), apodization=10, angles=[-45,45], range_angles=15)
    nu = kk / cutoff / pix_hifi
    ax.loglog(kk, power / 10.0**np.nanmean(np.log10(power[0:5])), label='Marginal', linewidth=2)
    
    im = gband_marginal[1].data[1:-1, 1:-1]
    kk, power = torchmfbd.util.azimuthal_power(im / np.nanmean(im), apodization=10, angles=[-45,45], range_angles=15)
    nu = kk / cutoff / pix_hifi
    ax.loglog(kk, power / 10.0**np.nanmean(np.log10(power[0:5])) , label='Speckle', linewidth=2)
    
    ax.legend()
    ax.set_ylim([1e-8, 5e1])
    ax.set_xlim([3e-3, 0.6])
    
    fig.supxlabel('Spatial frequency [1/pix]')
    fig.supylabel('Normalized power')
    
    ax.axvline(1.0 / (diff_hifi / pix_hifi), color='black')
    
    if save:
        pl.savefig('figs_marginal/power_hifi.pdf', dpi=300)



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


def crisp_noise(save=False):
    qs_8542_joint = fits.open('qs_8542/qs_8542_joint.fits')
    qs_8542_joint_0_01 = fits.open('qs_8542/qs_8542_joint_noise_0.01.fits')
    qs_8542_joint_0_1 = fits.open('qs_8542/qs_8542_joint_noise_0.1.fits')
    qs_8542_joint_0_3 = fits.open('qs_8542/qs_8542_joint_noise_0.3.fits')

    qs_8542_marginal = fits.open('qs_8542/qs_8542_marginal.fits')
    qs_8542_marginal_0_01 = fits.open('qs_8542/qs_8542_marginal_noise_0.01.fits')
    qs_8542_marginal_0_1 = fits.open('qs_8542/qs_8542_marginal_noise_0.1.fits')
    qs_8542_marginal_0_3 = fits.open('qs_8542/qs_8542_marginal_noise_0.3.fits')

    fig, ax = pl.subplots(nrows=3, ncols=4, figsize=(20, 15), tight_layout=True)

    ax[0, 0].imshow(qs_8542_joint[0].data[0, ...], origin='lower')
    ax[0, 0].set_title('QS 8542 - Frame')

    ax[0, 1].imshow(qs_8542_joint_0_01[0].data[0, ...], origin='lower')
    ax[0, 1].set_title('QS 8542 - Frame (noise=0.01)')

    ax[0, 2].imshow(qs_8542_joint_0_1[0].data[0, ...], origin='lower')
    ax[0, 2].set_title('QS 8542 - Frame (noise=0.1)')

    ax[0, 3].imshow(qs_8542_joint_0_3[0].data[0, ...], origin='lower')
    ax[0, 3].set_title('QS 8542 - Frame (noise=0.3)')


    ax[1, 0].imshow(qs_8542_joint[1].data[0, ...], origin='lower')
    ax[1, 0].set_title('QS 8542 - Joint')

    ax[1, 1].imshow(qs_8542_joint_0_01[1].data[0, ...], origin='lower')
    ax[1, 1].set_title('QS 8542 - Joint (noise=0.01)')

    ax[1, 2].imshow(qs_8542_joint_0_1[1].data[0, ...], origin='lower')
    ax[1, 2].set_title('QS 8542 - Joint (noise=0.1)')

    ax[1, 3].imshow(qs_8542_joint_0_3[1].data[0, ...], origin='lower')
    ax[1, 3].set_title('QS 8542 - Joint (noise=0.3)')

    ax[2, 0].imshow(qs_8542_marginal[1].data[0, ...], origin='lower')
    ax[2, 0].set_title('QS 8542 - Marginal')

    ax[2, 1].imshow(qs_8542_marginal_0_01[1].data[0, ...], origin='lower')
    ax[2, 1].set_title('QS 8542 - Marginal (noise=0.01)')

    ax[2, 2].imshow(qs_8542_marginal_0_1[1].data[0, ...], origin='lower')
    ax[2, 2].set_title('QS 8542 - Marginal (noise=0.1)')

    ax[2, 3].imshow(qs_8542_marginal_0_3[1].data[0, ...], origin='lower')
    ax[2, 3].set_title('QS 8542 - Marginal (noise=0.3)')

    if save:
        pl.savefig('figs/qs_8542_noise.pdf', dpi=300)


if __name__ == '__main__':

    pl.close('all')

    save = False

    # crisp(save)

    crisp_noise(save)
        
    # hifi(save)

    #power(save)
