import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter, MovieWriter
from tqdm import tqdm


def gen_movie(frames,             
              frames2=None, 
              filename='movie.gif',
              fps=1,
              deltat=300):
    
    """
    Generate an animated movie from a sequence of frames.

    Parameters:
        frames (torch.Tensor): A tensor containing the frames to be animated.
        frames2 (torch.Tensor, optional): An optional second tensor containing additional frames to be animated side-by-side. Default is None.
        filename (str, optional): The name of the output file. Default is 'movie.gif'.
        fps (int, optional): Frames per second for the output animation. Default is 1.
        deltat (int, optional): Time interval between frames in milliseconds. Default is 300.
    Returns:
        None
    """

    frames = frames.cpu().numpy()
    if frames2 is not None:
        frames2 = frames2.cpu().numpy()

    n_frames = frames.shape[0]

    # Create a figure and axis
    if frames2 is None:
        n_panels = 1
        fig, ax = plt.subplots()
    else:
        n_panels = 2
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

    t = tqdm(range(n_frames))

    def animate(i):

        if n_panels == 1:
            ax.clear()
            ax.imshow(frames[i])
        else:
            for j in range(n_panels):
                ax[j].clear()
            
            ax[0].imshow(frames[i])
            ax[1].imshow(frames2[i])

        t.update()
            
    ani = FuncAnimation(fig, 
                        animate, 
                        frames=n_frames,
                        interval=deltat, 
                        repeat=False)    

    if '.gif' in filename:
        ani.save(filename, 
             dpi=300,
             writer=PillowWriter(fps=fps))
    
    if '.mp4' in filename:    
        ani.save('movie.mp4',
             dpi=300,
             fps=fps)
    
    plt.close(fig)

    t.close()