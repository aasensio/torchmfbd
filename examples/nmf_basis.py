import torchmfbd

tmp = torchmfbd.Basis(n_pixel=128,
                 wavelength=8542.0,
                 diameter=100.0,
                 pix_size=0.059,
                 central_obs=0.0,
                 n_modes=150,
                 r0_min=5.0,
                 r0_max=30.0)
    
tmp.compute(type='nmf',n=150)