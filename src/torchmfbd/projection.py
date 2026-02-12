import numpy as np
import torchmfbd.zern as zern
import matplotlib.pyplot as pl
from tqdm import tqdm
from dict_hash import sha256
import glob
import pathlib
import logging

class ZernikeProjectionMatrix(object):
    def __init__(self):
        self.logger = logging.getLogger("projection ")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []
        ch = logging.StreamHandler()        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def zernikeProjectionMatrix(self, n_modes, XY, DTel, height, cm_per_pix=5.0):
        """Computes the matrix that relates the Zernike coefficients of an original metapupil with a footprint
        of smaller size inside the first one. The footprint is scaled and translated accordingly for 
        the positions and heights.
        
        b = M @ a
        
        with b the Zernike coefficients of the footprint in their local Zernike basis, M the transformation
        matrix, and a the original Zernike coefficients of the meta-pupil. This matrix assumes that no piston
        is present in the Zernike coefficients of the metapupil. However, piston is needed in the footprints.

        This routine carries out the numerical projection of the Zernike basis.
        
        Args:
            n_modes (int): maximum Noll index for the Zernikes        
            XY (float): XY coordinates (n_los, 2) of the patches to consider in arcsec
            DTel (float): Diameter of the telescope pupil [m].
            height (float): Height of the layer [km].
            cm_per_pix (float, optional): Centimeters per pixel in the metapupils and footprints. Default is 5.0.
        
        Returns:
            float: transformation matrix of size (n_los, n_modes, n_modes+1)
        """

        n_los = XY.shape[0]

        # Convert the XY coordinates to radians
        XY_rad = XY / 206265.0

        # Meters per pixel
        m_per_pix = cm_per_pix * 1e-2

        # Height of the layer [m]
        h_m = height * 1e3
        
        # Radius of the telescope pupil [m]    
        RTel = DTel / 2.0

        # Compute the radius of the meta-pupil
        # Find the largest radial distance in the XY array of positions of the patches    
        R = np.sqrt(XY_rad[:, 0]**2 + XY_rad[:, 1]**2)        
        RMetapupil = RTel + h_m * np.max(R)
        
        # Check if the matrix has been computed before
        identifier = {'n_modes': n_modes, 
                    'XY': XY.flatten(), 
                    'DTel': DTel, 
                    'h': height, 
                    'cm_per_pix': cm_per_pix}
        
        # Compute a hash code for the configuration
        hash_key = sha256(identifier)
        
        files = glob.glob('matrices/*.npy')    
        for file in files:
            if hash_key in file:
                self.logger.info(f'Loading matrix from {file}')
                M = np.load(file)            
                return M, 2.0 * RMetapupil        
        
        # Compute the number of pixels in the footprints of the telescope pupil and in the metapupil
        npix_Metapupil = int(2 * RMetapupil / m_per_pix)
        
        # Instantiate the Zernike machine
        Z_machine = zern.ZernikeNaive(mask=[])
        
        # Define the coordinates for evaluating the Zernike polynomials for the metapupil
        x = np.linspace(-RMetapupil, RMetapupil, npix_Metapupil)
        xx_Metapupil, yy_Metapupil = np.meshgrid(x, x)
        rho_Metapupil = np.sqrt(xx_Metapupil ** 2 + yy_Metapupil ** 2)
        theta_Metapupil = np.arctan2(yy_Metapupil, xx_Metapupil)
        mask_Metapupil = rho_Metapupil <= RMetapupil
                
        M = np.zeros((n_los, n_modes, n_modes))
        
        noll_Z_metapupil = 1 + np.arange(n_modes)
        noll_Z_footprint = 1 + np.arange(n_modes)
        
        for i in tqdm(range(n_modes)):

            n, m = zern.zernIndex(noll_Z_metapupil[i])
            ZMetapupil = Z_machine.Z_nm(n, m, rho_Metapupil / RMetapupil, theta_Metapupil, True, 'Jacobi') * mask_Metapupil

            loop = 0
            
            for j in range(n_los):
                xfp = h_m * XY_rad[j, 0]
                yfp = h_m * XY_rad[j, 1]

                # Define the mask for the footprint
                mask = (xx_Metapupil - xfp)**2 + (yy_Metapupil - yfp)**2 <= RTel**2
                
                if j == 0 and i == 0:
                    Z_all = []
                    Z_computed = -np.ones((n_los, n_modes+1), dtype=int)
                                                
                # Evaluate the overlap of the Zernike polynomials on the footprint and the metapupil
                for k in range(n_modes):                
                    if Z_computed[j, k] > 0:                        
                        ZTel = Z_all[Z_computed[j, k]]
                    else:                                                

                        # Compute the xy coordinates of the footprint
                        xx_Tel = xx_Metapupil[mask] - xfp
                        yy_Tel = yy_Metapupil[mask] - yfp
                        rho_Tel = np.sqrt(xx_Tel ** 2 + yy_Tel ** 2)
                        theta_Tel = np.arctan2(yy_Tel, xx_Tel)
                        mask_Tel = rho_Tel <= RTel            
                        n, m = zern.zernIndex(noll_Z_footprint[k])
                        ZTel = Z_machine.Z_nm(n, m, rho_Tel / RTel, theta_Tel, True, 'Jacobi') * mask_Tel
                        Z_all.append(ZTel)
                        Z_computed[j, k] = loop
                        loop += 1
                    
                    M[j, i, k] = np.sum(ZMetapupil[mask] * ZTel) / np.sum(ZTel**2)

        self.logger.info(f'Saving Zernike projection matrix on matrices/M_{hash_key}.npy')
        p = pathlib.Path('matrices/')
        p.mkdir(parents=True, exist_ok=True)
        np.save(f'matrices/M_{hash_key}.npy', M)
                                        
        return M, 2.0 * RMetapupil

    def _testProjectionMatrix(self, n_modes, XY, DTel, height, M, cm_per_pix=5.0, verbose=True):
        """Computes the matrix that relates the Zernike coefficients of an original meta-pupil with a footprint
        of smaller size inside the first one. The footprint is scaled and translated.
        
        b = M @ a
        
        with b the Zernike coefficients of the footprint in their local Zernike basis, M the transformation
        matrix, and a the original Zernike coefficients of the meta-pupil.

        This routine carries out the numerical projection of the Zernike basis.
        
        Args:
            nMax (int): maximum Noll index for the Zernikes
            XY (float): XY coordinates (n_los, 2) of the patches to consider        
            DTel (float): Diameter of the telescope pupil [m].
            height (float): Height of the layer [km].
            M (numpy.ndarray): Transformation matrix.
            cm_per_pix (float, optional): Centimeters per pixel. Default is 5.0.
            verbose (bool, optional): If True, print verbose output. Default is True.
            
            verbose (bool, optional): verbose
            includePiston (bool, optional): include the piston coefficient
        
        Returns:
            float: transformation matrix
        
        Deleted Parameters:
            alpha (float): rotation angle of the center of the footprint [radians]
            process (bool, optional): Description
        """

        n_los = XY.shape[0]

        XY_rad = XY / 206265.0

        # Meters per pixel
        m_per_pix = cm_per_pix * 1e-2

        # Height of the layer [m]
        h_m = height * 1e3
        
        # Radius of the telescope pupil [m]    
        RTel = DTel / 2.0

        # Compute the radius of the meta-pupil
        # Find the largest radial distance in the XY array of positions of the patches    
        R = np.sqrt(XY_rad[:, 0]**2 + XY_rad[:, 1]**2)        
        RMetapupil = RTel + h_m * np.max(R)

        # Compute the number of pixels in the footprints of the telescope pupil and in the metapupil
        npix_Tel = int(DTel / m_per_pix)
        npix_Metapupil = int(2 * RMetapupil / m_per_pix)
        
        # Instantiate the Zernike machine
        Z_machine = zern.ZernikeNaive(mask=[])
        
        # Define the coordinates for evaluating the Zernike polynomials    
        # For the metapupil
        x = np.linspace(-RMetapupil, RMetapupil, npix_Metapupil)
        xx_Metapupil, yy_Metapupil = np.meshgrid(x, x)
        rho_Metapupil = np.sqrt(xx_Metapupil ** 2 + yy_Metapupil ** 2)
        theta_Metapupil = np.arctan2(yy_Metapupil, xx_Metapupil)
        mask_Metapupil = rho_Metapupil <= RMetapupil

        x = np.linspace(-RTel, RTel, npix_Tel)
        xx_Tel, yy_Tel = np.meshgrid(x, x)
        rho_Tel = np.sqrt(xx_Tel ** 2 + yy_Tel ** 2)
        theta_Tel = np.arctan2(yy_Tel, xx_Tel)
        mask_Tel = rho_Tel <= RTel
                    
        noll_Z_metapupil = 2 + np.arange(n_modes)
        noll_Z_footprint = 1 + np.arange(n_modes+1)

        modes = np.random.randn(n_modes)

        wavefront = np.zeros((npix_Metapupil, npix_Metapupil))    

        for i in tqdm(range(n_modes)):

            n, m = zern.zernIndex(noll_Z_metapupil[i])
            ZMetapupil = Z_machine.Z_nm(n, m, rho_Metapupil / RMetapupil, theta_Metapupil, True, 'Jacobi') * mask_Metapupil

            wavefront += modes[i] * ZMetapupil

        fig, ax = pl.subplots()
        im = ax.imshow(wavefront, vmin=-4, vmax=4)
        pl.colorbar(im, ax=ax)

        fig, ax = pl.subplots(nrows=4, ncols=5, figsize=(20, 15))
                    
        for j in range(5):#n_los):
            xfp = h_m * XY_rad[j, 0]
            yfp = h_m * XY_rad[j, 1]

            # Define the mask for the footprint
            mask = (xx_Metapupil - xfp)**2 + (yy_Metapupil - yfp)**2 <= RTel**2

            wavefront_new = np.copy(wavefront)
            wavefront_new[~mask] = 0.0

            xa = np.argmin(np.abs((xx_Metapupil[0,:]-xfp) + RTel))
            xb = np.argmin(np.abs((xx_Metapupil[0,:]-xfp) - RTel))

            x0 = np.min([xa, xb])
            x1 = x0 + npix_Tel

            ya = np.argmin(np.abs((yy_Metapupil[:,0]-yfp) + RTel))
            yb = np.argmin(np.abs((yy_Metapupil[:,0]-yfp) - RTel))

            y0 = np.min([ya, yb])
            y1 = y0 + npix_Tel
            
            modes_footprint = M[j, :, :].T @ modes
            
            wavefront_footprint = np.zeros((npix_Tel, npix_Tel))

            for k in range(n_modes+1):

                n, m = zern.zernIndex(noll_Z_footprint[k])
                Z = Z_machine.Z_nm(n, m, rho_Tel / RTel, theta_Tel, True, 'Jacobi') * mask_Tel

                wavefront_footprint += modes_footprint[k] * Z
            
            im = ax[0, j].imshow(wavefront_new, vmin=-4, vmax=4)
            pl.colorbar(im, ax=ax[0, j])
            im = ax[1, j].imshow(wavefront_new[y0:y1, x0:x1], vmin=-4, vmax=4)
            pl.colorbar(im, ax=ax[1, j])
            im = ax[2, j].imshow(wavefront_footprint, vmin=-4, vmax=4)
            pl.colorbar(im, ax=ax[2, j])
            residual = wavefront_new[y0:y1, x0:x1] - wavefront_footprint
            residual2 = wavefront_new[y0:y1, x0:x1] / wavefront_footprint
            im = ax[3, j].imshow(residual)
            pl.colorbar(im, ax=ax[3, j])
            print(j, np.mean(residual2[20:60, 20:60]))
            
            
            # breakpoint()

            # Compute the xy coordinates of the footprint
            # xx_Tel = xx_Metapupil[mask] - xfp
            # yy_Tel = yy_Metapupil[mask] - yfp
            # rho_Tel = np.sqrt(xx_Tel ** 2 + yy_Tel ** 2)
            # theta_Tel = np.arctan2(yy_Tel, xx_Tel)
                                                    
        return

    def plotMetapupils(self, XY, DTel, h):
        """Plot the pupils
                
        """

        nh = len(h)

        f, ax = pl.subplots(nrows=1, ncols=nh, figsize=(5*nh, 5))

        XY_rad = XY / 206265.0
        
        RTel = DTel / 2.0
        
        for loop in range(nh):
                        
            n = XY.shape[0]
                                    
            R = np.sqrt(XY_rad[:, 0]**2 + XY_rad[:, 1]**2)        
            RMetapupil = RTel + 1e3 * h[loop] * np.max(R)

            for j in range(n):
                xfp = 1e3 * h[loop] * XY_rad[j, 0]
                yfp = 1e3 * h[loop] * XY_rad[j, 1]
                    
                circle = pl.Circle((0,0), RMetapupil, fill=False, linewidth=4, axes=ax[loop], color='C2')
                ax[loop].add_artist(circle)

                circle = pl.Circle((0,0), RTel, fill=False, linewidth=4, axes=ax[loop], color='C1')
                ax[loop].add_artist(circle)

                ax[loop].set_xlim([-RMetapupil, RMetapupil])
                ax[loop].set_ylim([-RMetapupil, RMetapupil])

                                    
                circle = pl.Circle((xfp, yfp), RTel, fill=False, axes=ax[loop], linewidth=2, color='C0')
                ax[loop].add_artist(circle)

if __name__ == '__main__':
    # zernikeProjectionMatrixNumerical(44, 0.5, 0.4, 0.0, radius=128, verbose=True, includePiston=False)

    pl.close('all')
    # plotPupils(4.0, [0.0, 5.0e3, 10e3, 15e3])

    n = 3
    x = np.linspace(-60.0, 60.0, n) / 206265.0
    y = np.linspace(-60.0, 60.0, n) / 206265.0
    X, Y = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    XY = np.concatenate([X[:, None], Y[:, None]], axis=1)
                        
    n_modes = 44
    DTel = 4.0  # m
    height = 10.0  # km

    z = ZernikeProjectionMatrix()

    M = z.zernikeProjectionMatrix(n_modes, XY, DTel, height, cm_per_pix=5.0)
    z._testProjectionMatrix(n_modes, XY, DTel, height, M, cm_per_pix=5.0)