import numpy as np
from astropy.io import fits
from tqdm import tqdm
try:
    DESTRETCH = True
    import GridMatch as GM
except:
    DESTRETCH = False


def readsst(root, label, cam=0, lam=4, mod=0, seq=[0, 5], xrange=None, yrange=None, instrument='CRISP', destretch=False):

    ns = seq[1] - seq[0]

    loop = 0

    tiles = [8,16,32,48]
    clips = [15,10,5,3]
    nthreads = 16

    if destretch and not DESTRETCH:
        print('Destretching is not available. Please install GridMatch.')        

    for i in tqdm(range(seq[0], seq[1])):
        filename_wb = f'{root}/wb_{label}_{i:05d}.fits'
        filename_nb = f'{root}/nb_{label}_{i:05d}.fits'
        
        f_wb = fits.open(filename_wb)
        f_nb = fits.open(filename_nb)

        if (instrument == 'CHROMIS'):
            filename_db = f'{root}/db_{label}_{i:05d}.fits'
            f_db = fits.open(filename_db)

        if (instrument == 'CRISP'):
            ncam, nf, nl, nmod, nx, ny = f_nb[0].data.shape
        if (instrument == 'CHROMIS'):
            nf, nl, nx, ny = f_nb[0].data.shape

        if (i == seq[0]):
            if (xrange is None):
                xrange = [0, nx]
            if (yrange is None):
                yrange = [0, ny]

            nx = xrange[1] - xrange[0]
            ny = yrange[1] - yrange[0]

            wb = np.zeros((ns, nf, nx, ny), dtype=np.float32)
            nb = np.zeros((ns, nf, nx, ny), dtype=np.float32)
            if (instrument == 'CHROMIS'):
                db = np.zeros((ns, nf, nx, ny), dtype=np.float32)            

        if (instrument == 'CRISP'):
            wb[loop, :, :, :] = f_wb[0].data[:, lam, mod, xrange[0]:xrange[1], yrange[0]:yrange[1]]
            nb[loop, :, :, :] = f_nb[0].data[cam, :, lam, mod, xrange[0]:xrange[1], yrange[0]:yrange[1]]

        if (instrument == 'CHROMIS'):
            wb[loop, :, :, :] = f_wb[0].data[:, lam, xrange[0]:xrange[1], yrange[0]:yrange[1]]
            nb[loop, :, :, :] = f_nb[0].data[:, lam, xrange[0]:xrange[1], yrange[0]:yrange[1]]
            db[loop, :, :, :] = f_db[0].data[:, lam, xrange[0]:xrange[1], yrange[0]:yrange[1]]
        
        if (destretch and DESTRETCH):
            cor, wb[loop, ...] = GM.DSGridNestBurst(wb[loop, ...].astype('float64'), tiles, clips, nthreads = nthreads, apply_correction = True)
                    
            # Apply the destretching to the NB image
            for i in range(nf):
                nb[loop, i, :, :] = GM.Stretch(nb[loop, i, :, :].astype('float64'), cor[i], nthreads= nthreads)

        f_wb.close()
        f_nb.close()

        if (instrument == 'CHROMIS'):
            f_db.close()

        loop += 1

    if (instrument == 'CHROMIS'):
        return wb, nb, db
    else:
        return wb, nb