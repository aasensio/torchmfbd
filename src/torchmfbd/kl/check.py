import numpy as np
import scipy.special as sp
import zern
import torch
import scipy.linalg as la
import time

def _even(x):
    return x%2 == 0

def _zernike_parity( j, jp):
    return _even(j-jp)


if __name__ == '__main__':

    Q = 50
    n_modes = (Q+1)*(Q+2)//2
    method = 'svd'

    print(n_modes)

    covariance = np.zeros((n_modes, n_modes))
    for j in range(n_modes):

        n, m = zern.zernIndex(j+1)

        for jpr in range(n_modes):
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

    covariance[0, :] = 0.0
    covariance[:, 0] = 0.0
    covariance[0, 0] = 1.0

    # with np.printoptions(precision=4, suppress=True):
        # print(covariance[1:9, 1:9])

    tinit = time.time()
    if method == 'svd':
        uu, ss, vh = np.linalg.svd(covariance, hermitian=True)
        evec = vh
    elif method == 'eig':
        eval, evec = np.linalg.eigh(covariance)
    elif method == 'svd_scipy':        
        uu, ss, vh = la.svd(covariance)
        evec = vh

    print('Elapsed time: ', time.time() - tinit)

    tmp = np.load('kl_data.npy')

    noll_KL = tmp[:, 0]
    noll_Z = tmp[:, 1]
    cz = tmp[:, 2]

    for i in range(4):
        ind = noll_KL == i+1

        print('Noll index: ', i+1)        
        print('   Zernike index: ', noll_Z[ind]-1)
        print('   CZ1: ', cz[ind])

        order = np.argsort(np.abs(evec[i, :]))[::-1]

        ind2 = np.where(np.abs(evec[i, order]) > 1e-5)[0]        
        
        print('   Zernike index2: ', order[ind2])
        print('   CZ2: ', evec[i, order][ind2])