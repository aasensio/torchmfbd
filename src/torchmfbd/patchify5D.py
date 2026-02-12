import torch.nn.functional as F
import torch
from einops import rearrange

class Patchify5D(object):
    def __init__(self):
        super().__init__()
        # self.p = patch_size
        # self.unfold = torch.nn.Unfold(kernel_size=patch_size, stride=stride_size)
        # self.fold = torch.nn.Fold(kernel_size=patch_size, stride=stride_size)

    def patchify(self, x, patch_size=56, stride_size=56, flatten_sequences=True):
        """
        Splits the input tensor into patches.
        Args:
            x (torch.Tensor): Input tensor of shape (n_scans, n_obj, n_frames, nx, ny).
            patch_size (int, optional): Size of each patch. Default is 56.
            stride_size (int, optional): Stride size for patch extraction. Default is 56.
        Returns:
            torch.Tensor: Tensor containing the patches with shape (n_scans, L, n_obj, n_frames, patch_size, patch_size),
                          where L is the number of patches extracted.
        """
        
        if x.ndim == 4:
            x = x.unsqueeze(2)
                    
        self.n_scans, self.n_obj, self.n_frames, self.nx, self.ny = x.shape

        self.output_size = (self.nx, self.ny)
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.flatten_sequences = flatten_sequences

        # This masks is used to weight the patches when unpatchifying to allow for overlapping patches
        self.mask = torch.ones_like(x).to(x.device)
        
        x = rearrange(x, 'n o f x y -> (n o) f x y')
        self.mask = rearrange(self.mask, 'n o f x y -> (n o) f x y')        
                
        x = F.unfold(x, kernel_size=self.patch_size, stride=self.stride_size)
        self.mask = F.unfold(self.mask, kernel_size=self.patch_size, stride=self.stride_size)

        self.n_patches = x.shape[-1]
        # x -> B (c*p*p) L
        
        # Reshaping into the shape we want
        if self.flatten_sequences:            
            a = rearrange(x, '(n o) (f x y) L -> (n L) o f x y', x=self.patch_size, y=self.patch_size, n=self.n_scans, o=self.n_obj)
            self.mask = rearrange(self.mask, '(n o) (f x y) L -> (n L) o f x y', x=self.patch_size, y=self.patch_size, n=self.n_scans, o=self.n_obj)
        else:
            a = rearrange(x, '(n o) (f x y) L -> n L o f x y', x=self.patch_size, y=self.patch_size, n=self.n_scans, o=self.n_obj)
            self.mask = rearrange(self.mask, '(n o) (f x y) L -> n L o f x y', x=self.patch_size, y=self.patch_size, n=self.n_scans, o=self.n_obj)
                
        return a
    
    def unpatchify(self, x, apodization=0):
        """
        Reconstructs the original image from patches.
        Args:
            x (torch.Tensor): The input tensor containing image patches with shape
                              (n, L, o, f, x, y), where:
                              - n: number of scans
                              - L: number of patches
                              - o: number of objects
                              - f: number of frames (optional)
                              - x, y: patch dimensions
            apodization (int, optional): Number of pixels to apodize at the edges of the image. Default is 0.
        Returns:
            torch.Tensor: The reconstructed image tensor with shape 
                          (n, o, f, x, y), where:
                          - n: number of scans
                          - o: number of objects
                          - f: number of features (optional)
                          - x, y: image dimensions
        """

        xndim = x.ndim
        if xndim == 4:
            if self.flatten_sequences:
                x = x.unsqueeze(2)
            else:
                x = x.unsqueeze(3)
            mask = self.mask[:, :, 0:1, :, :]
        else:
            mask = self.mask

        mask = mask.to(x.device)

        if apodization > 0:
            x = x[:, :, :, apodization:-apodization, apodization:-apodization]
            mask = mask[:, :, :, apodization:-apodization, apodization:-apodization]

        output_size = (self.output_size[0] - 2*apodization, self.output_size[1] - 2*apodization)
        patch_size = self.patch_size - 2*apodization
        stride_size = self.stride_size
                
        if self.flatten_sequences:
            x = rearrange(x, '(n L) o f x y -> (n o) (f x y) L', n=self.n_scans, L=self.n_patches)
            mask = rearrange(mask, '(n L) o f x y -> (n o) (f x y) L', n=self.n_scans, L=self.n_patches)
        else:
            x = rearrange(x, 'n L o f x y -> (n o) (f x y) L')
            mask = rearrange(mask, 'n L o f x y -> (n o) (f x y) L')
        
        x = F.fold(x, output_size=output_size, kernel_size=patch_size, stride=stride_size)
        mask = F.fold(mask, output_size=output_size, kernel_size=patch_size, stride=stride_size)

        x = rearrange(x, '(n o) f x y -> n o f x y', n=self.n_scans, o=self.n_obj)
        mask = rearrange(mask, '(n o) f x y -> n o f x y', n=self.n_scans, o=self.n_obj)
        
        final = x / mask

        if xndim == 4:
            final = final.squeeze(2)
                
        return final
    
if __name__ == '__main__':
    n_scans = 1
    n_obj = 2
    n_frames = 12
    n_x = 80
    n_y = 80
    
    frames = torch.randn(n_scans, n_obj, n_frames, n_x, n_y)

    p = Patchify()

    patches = p.patchify(frames, patch_size=20, stride_size=10)

    out = p.unpatchify(patches)