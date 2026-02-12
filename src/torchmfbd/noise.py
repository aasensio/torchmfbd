from torchmfbd.noise_svd import noise_estimation
import torch

def compute_noise(frames):
    """
    Compute the noise characteristics of the images.
    Parameters:
    -----------
    frames : numpy.ndarray
        A 5D array of shape (n_scans, n_frames, nx, ny) representing the image frames.
    Returns:
    --------
    sigma : numpy.ndarray
        A 4D array of shape (n_obj, n_scans, n_frames, nx*ny) representing the estimated noise characteristics.
    """
    
    n_scans, n_frames, nx, ny = frames.shape
    
    sigma = noise_estimation(frames.numpy())

    sigma = torch.tensor(sigma.astype('float32')).to(frames.device)
    
    return sigma