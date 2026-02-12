import numpy as np
from torchmfbd.zern import zernIndex, ZernikeNaive
import matplotlib.pyplot as pl
from tqdm import tqdm
import scipy.special as sp
import pkgutil
from io import BytesIO
import logging

def _even(x):
    return x%2 == 0

def _zernike_parity( j, jp):
    return _even(j-jp)

class KL(object):

    def __init__(self):        
        f = pkgutil.get_data(__package__, 'kl/kl_data_npy')
        tmp = np.load(BytesIO(f))        
        self.noll_KL = tmp[:,0].astype('int')
        self.noll_Z = tmp[:,1].astype('int')
        self.cz = tmp[:,2]
        f = pkgutil.get_data(__package__, 'kl/KL_variance_npy')
        self.KL_variance = np.load(BytesIO(f))

        self.logger = logging.getLogger("modes")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []
        ch = logging.StreamHandler()        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.Z_machine = ZernikeNaive(mask=[])

    def kl(self, rho, theta, j):
        """
        Compute the KL mode j at the coordinates rho, theta
        
        Parameters
        ----------
        rho : float array
            Radial coordinates
        theta : float array
            Azimuthal coordinates
        j : int
            Mode number
        
        Returns
        -------
        float array
            KL mode
        """

        indx = np.where(self.noll_KL == j)[0]
        print(indx)

        KL = np.zeros_like(rho)

        for i in range(len(indx)):                
            jz = self.noll_Z[indx[i]]
            cz = self.cz[indx[i]]

            n, m = zernIndex(jz)
            
            Z = self.Z_machine.Z_nm(n, m, rho, theta, True, 'Jacobi')
            KL += cz * Z
                    
        return KL

    def precalculate(self, npix_image, n_modes_max, overfill=1.0):
        """
        Precalculate KL modes. We skip the first mode, which is just the
        aperture. The second and third mode (first and second one in the return)
        are tip-tilt using standard Zernike modes. The rest are KL modes
        obtained by the diagonalization of the Kolmogorov Noll's matrix
        
        Parameters
        ----------
        npix_image : int
            Number of pixels on the pupil plane
        
        Returns
        -------
        float array
            KL modes
        """
        
        self.npix_image = npix_image
        self.n_modes_max = n_modes_max
        
        Z_machine = ZernikeNaive(mask=[])
        x = np.linspace(-1, 1, npix_image)
        xx, yy = np.meshgrid(x, x)
        rho = overfill * np.sqrt(xx ** 2 + yy ** 2)
        theta = np.arctan2(yy, xx)
        aperture_mask = rho <= 1.0

        # Precalculate Zernike modes
        max_jz = 0
        for mode in range(self.n_modes_max):
            
            # Modes start at 1
            j = mode + 1
                        
            indx = np.where(self.noll_KL == j)[0]

            jz = self.noll_Z[indx]

            max_jz = max(max_jz, np.max(jz))

        self.logger.info("Computing Zernike modes")
        zernikes = np.zeros((max_jz, self.npix_image, self.npix_image))
        for j in tqdm(range(max_jz)):
            n, m = zernIndex(j + 1)
            zernikes[j, :, :] = Z_machine.Z_nm(n, m, rho, theta, True, 'Jacobi')

        self.KL = np.zeros((self.n_modes_max + 1, self.npix_image, self.npix_image))
        self.varKL = np.zeros((self.n_modes_max))
        
        self.logger.info("Computing KL modes")
        for mode in tqdm(range(self.n_modes_max)):
            
            # Modes start at 1
            j = mode + 1
                        
            indx = np.where(self.noll_KL == j)[0]

            for i in range(len(indx)):                
                jz = self.noll_Z[indx[i]]
                cz = self.cz[indx[i]]

                n, m = zernIndex(jz)
                
                # Z = Z_machine.Z_nm(n, m, rho, theta, True, 'Jacobi')
                # self.KL[mode,:,:] += cz * Z * aperture_mask

                self.KL[mode,:,:] += cz * zernikes[jz - 1, :, :] * aperture_mask
            
            self.varKL[mode] = self.KL_variance[j-1]

        # Now remove the piston mode that we never use
        # The variances do not include piston
        self.KL = self.KL[1:, :, :]
        
        return self.KL

    def precalculate_covariance(self, npix_image, n_modes_max, first_noll=1, overfill=1.0):
        """
        Precalculate KL modes. We skip the first mode, which is just the
        aperture. The second and third mode (first and second one in the return)
        are tip-tilt using standard Zernike modes. The rest are KL modes
        obtained by the diagonalization of the Kolmogorov Noll's matrix
        
        Parameters
        ----------
        npix_image : int
            Number of pixels on the pupil plane
        n_modes_max : int
            Maximum number of nodes to consider
        first_noll : int
            First Noll index to consider. j=1 is the aperture. j=2/3 are the tip-tilts
        
        Returns
        -------
        float array
            KL modes
        """

        self.npix_image = npix_image
        self.first_noll = first_noll - 1
        self.n_modes_max = n_modes_max + first_noll

        print("Computing Kolmogorov covariance...")
        covariance = np.zeros((self.n_modes_max, self.n_modes_max))
        for j in range(self.n_modes_max):
            n, m = zern.zernIndex(j+1)

            for jpr in range(self.n_modes_max):
                npr, mpr = zern.zernIndex(jpr+1)
                
                deltaz = (m == mpr) and (_zernike_parity(j, jpr) or m == 0)
                
                if (deltaz):                
                    phase = (-1.0)**(0.5*(n+npr-2*m))
                    t1 = np.sqrt((n+1)*(npr+1)) 
                    t2 = sp.gamma(14.0/3.0) * sp.gamma(11.0/6.0)**2 * (24.0/5.0*sp.gamma(6.0/5.0))**(5.0/6.0) / (2.0*np.pi**2)
                    
                    Kzz = t2 * t1 * phase
                    
                    t1 = sp.gamma(0.5*(n+npr-5.0/3.0))
                    t2 = sp.gamma(0.5*(n-npr+17.0/3.0)) * sp.gamma(0.5*(npr-n+17.0/3.0)) * sp.gamma(0.5*(n+npr+23.0/3.0))
                    covariance[jpr,j] = Kzz * t1 / t2

        covariance[0,:] = 0.0
        covariance[:,0] = 0.0
        covariance[0,0] = 1.0

        covariance  = covariance[first_noll:, first_noll:]
        
        print("Diagonalizing Kolmogorov covariance...")
        uu, ss, vh = np.linalg.svd(covariance)

        # vh[np.abs(vh) < 1e-15] = 0.0

        print("Computing KL modes...")
        Z_machine = zern.ZernikeNaive(mask=[])
        x = np.linspace(-1, 1, npix_image)
        xx, yy = np.meshgrid(x, x)
        rho = overfill * np.sqrt(xx ** 2 + yy ** 2)
        theta = np.arctan2(yy, xx)
        aperture_mask = rho <= 1.0

        self.KL = np.zeros((self.n_modes_max, self.npix_image, self.npix_image))

        noll_Z = first_noll + np.arange(self.n_modes_max)

        for mode in tqdm(range(self.n_modes_max)):
            
            cz = vh[mode,:]
            ind = np.where(cz != 0)[0]

            print(mode)
            
            for i in range(len(ind)):
                jz = noll_Z[ind[i]]                
                coeff = cz[ind[i]]
                
                n, m = zern.zernIndex(jz)

                print(jz, n, m, coeff)
                Z = Z_machine.Z_nm(n, m, rho, theta, True, 'Jacobi')
                self.KL[mode,:,:] += coeff * Z * aperture_mask
        
        self.KL = self.KL[self.first_noll+1:,:,:]
        
        return self.KL

if (__name__ == '__main__'):

    tmp = KL()

    tmp.kl(0.5, 0.5, 2)
    
    # kl = tmp.precalculate_covariance(npix_image=128, n_modes_max=6, first_noll=2)

    # f, ax = pl.subplots(nrows=4, ncols=4)
    # for i in range(16):
    #     ax.flat[i].imshow(tmp.KL[i, :, :])
    # pl.show()

    # mat = np.zeros((20,20))
    # for i in range(20): 
    #     for j in range(20): 
    #         mat[i,j] = np.sum(tmp.KL[i,:,:]*tmp.KL[j,:,:])

    # #pl.imshow(np.log(np.abs(mat)))
    # #pl.show()
