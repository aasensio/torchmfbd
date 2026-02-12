import numpy as np
import torch
import torch.nn.functional as F
from torchmfbd.warp import warp, warp_affine
import torchmfbd.util as util
from tqdm import tqdm
from einops import rearrange, repeat
from collections import OrderedDict

def destretch(frames,             
              ngrid=32, 
              lr=0.01,
              reference_frame=0,
              border=10,
              n_iterations=20,
              lambda_tt=0.1,
              mode='bilinear'
              ):
    
    """
    Perform image destretching on a sequence of frames using gradient-based optimization.
    It optimizes the optical flow in the field-of-view to align the frames to a reference frame.
    To this end, it uses the correlation between the reference frame and the warped frames
    as defined in "Parametric Image Alignment Using Enhanced Correlation Coefficient Maximization"
    by Georgios D. Evangelidis and Emmanouil Z. Psarakis.
    
    Args:
        frames (torch.Tensor): Input tensor of shape (n_seq, n_o, n_f, n_x, n_y) representing the sequence of frames.
        ngrid (int, optional): Grid size for the tip-tilt estimation. Default is 32.
        lr (float, optional): Learning rate for the optimizer. Default is 0.01.
        reference_frame (int, optional): Index of the reference frame to which other frames are aligned. Default is 0.
        border (int, optional): Border size to exclude from the loss computation. Default is 10.
        n_iterations (int, optional): Number of optimization iterations. Default is 20.
        lambda_tt (float, optional): Regularization parameter for the tip-tilt smoothness. Default is 0.1.
        mode (str, optional): Interpolation mode for the warping. Default is 'bilinear' ('nearest'/'bilinear').        
    Returns:
        tuple: A tuple containing:
            - warped (torch.Tensor): Warped frames after destretching.
            - tt (torch.Tensor): Estimated tip-tilt values.
    """
    ndim = len(frames.shape)
    if ndim == 3:        
        frames = frames[None, None, ...]

    device = frames.device
    n_seq, n_o, n_f, n_x, n_y = frames.shape

    finite_difference = util.FiniteDifference().to(frames.device)

    tiptilt = torch.zeros((n_seq, n_f, 2, ngrid, ngrid), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([tiptilt], lr=lr)
    
    # Reference frame
    ir = frames[:, :, reference_frame:reference_frame + 1, :, :]
    ir = ir - torch.mean(ir, keepdim=True, dim=(-1, -2))

    t = tqdm(range(n_iterations))

    im = rearrange(frames, 'nb no nf nx ny -> (nb nf) no nx ny')

    for loop in t:

        optimizer.zero_grad()
        
        tt = rearrange(tiptilt, 'nb nf d nx ny -> (nb nf) d nx ny')
        tt = F.interpolate(tt, size=(n_x, n_y), mode='bilinear')
        warped = warp(im, tt, mode='bilinear')
        
        warped = rearrange(warped, '(nb nf) no nx ny -> nb no nf nx ny', nb=n_seq)

        iw = warped - torch.mean(warped, keepdim=True, dim=(-1, -2))

        t1 = ir / torch.linalg.norm(ir, dim=(-1, -2), keepdim=True)
        t2 = iw / torch.linalg.norm(iw, dim=(-1, -2), keepdim=True)

        if border > 0:
            t1 = t1[..., border:-border, border:-border]
            t2 = t2[..., border:-border, border:-border]

        loss_corr = torch.linalg.norm(t1 - t2, dim=(-1,-2))
        loss_corr = torch.mean(loss_corr)

        # Regularization
        regularization = torch.tensor(0.0, device=device)
        if lambda_tt > 0.0:
            tmp = rearrange(tiptilt, 'nb nf nm nx ny -> (nb nf) nm nx ny')
            regularization = lambda_tt * torch.mean(finite_difference(tmp)**2)            
                
        loss = loss_corr + regularization
        
        loss.backward()

        optimizer.step()

        tmp = OrderedDict()
        tmp['Lcorr'] = f'{loss_corr.item():8.6f}'
        tmp['R'] = f'{regularization.item():8.6f}'
        tmp['L'] = f'{loss.item():8.6f}'
        
        t.set_postfix(ordered_dict = tmp)

    if mode == 'nearest':
        warped = warp(im, tt, mode='nearest')
        warped = rearrange(warped, '(nb nf) no nx ny -> nb no nf nx ny', nb=n_seq)

    if ndim == 3:
        warped = warped[0, 0, ...]

    return warped.detach(), tt.detach()


def apply_destretch(frames, tt, mode='bilinear'):

    n_seq, n_o, n_f, n_x, n_y = frames.shape
    
    im = rearrange(frames, 'nb no nf nx ny -> (nb nf) no nx ny')
    warped = warp(im, tt, mode=mode)

    warped = rearrange(warped, '(nb nf) no nx ny -> nb no nf nx ny', nb=n_seq)

    return warped


def align(frames,              
              lr=0.01,              
              border=10,
              n_iterations=20,              
              mode='bilinear'              
              ):
    
    """
    Perform image alignment between two images using gradient-based optimization.
    It optimizes the affine transformation matrix to align the second frame to the first frame, which is used as reference.
    To this end, it uses the correlation between the reference frame and the warped frames
    as defined in "Parametric Image Alignment Using Enhanced Correlation Coefficient Maximization"
    by Georgios D. Evangelidis and Emmanouil Z. Psarakis.
    
    Args:
        frames (torch.Tensor): Input tensor of shape (n_f, n_x, n_y) representing the sequence of frames.        
        lr (float, optional): Learning rate for the optimizer. Default is 0.01.        
        border (int, optional): Border size to exclude from the loss computation. Default is 10.
        n_iterations (int, optional): Number of optimization iterations. Default is 20.        
        mode (str, optional): Interpolation mode for the warping. Default is 'bilinear' ('nearest'/'bilinear').        
    Returns:
        tuple: A tuple containing:
            - warped (torch.Tensor): Warped frames after alignment.
            - tt (torch.Tensor): Estimated affine matrix.
    """
    
    device = frames.device
    n_f, n_x, n_y = frames.shape

    finite_difference = util.FiniteDifference().to(frames.device)
    
    affine = np.zeros((1, 2, 3), dtype=np.float32)
    affine[:, 0, 0] = 1.0
    affine[:, 1, 1] = 1.0
    affine = torch.tensor(affine, device=device, requires_grad=True)        
    optimizer = torch.optim.Adam([affine], lr=lr)

    # Reference frame
    ir = frames[0:1, :, :]
    im = frames[1:2, :, :]

    ir = ir - torch.mean(ir, keepdim=True, dim=(-1, -2))
    
    t = tqdm(range(n_iterations))
    
    for loop in t:

        optimizer.zero_grad()
                
        warped = warp_affine(im[None, ...], affine, mode='bilinear')[0, ...]
        
        iw = warped - torch.mean(warped, keepdim=True, dim=(-1, -2))

        t1 = ir / torch.linalg.norm(ir, dim=(-1, -2), keepdim=True)
        t2 = iw / torch.linalg.norm(iw, dim=(-1, -2), keepdim=True)

        if border > 0:
            t1 = t1[..., border:-border, border:-border]
            t2 = t2[..., border:-border, border:-border]

        loss = torch.linalg.norm(t1 - t2, dim=(-1,-2))
        loss = torch.mean(loss)
                        
        loss.backward()

        optimizer.step()

        tmp = OrderedDict()        
        tmp['L'] = f'{loss.item():8.6f}'
        
        t.set_postfix(ordered_dict = tmp)

    if mode == 'nearest':
        warped = warp(im, tt, mode='nearest')
        warped = rearrange(warped, '(nb nf) no nx ny -> nb no nf nx ny', nb=n_seq)

    return warped.detach(), affine.detach()


def apply_align(frames, affine, mode='bilinear'):
        
    warped = warp_affine(frames, affine, mode=mode)

    return warped
