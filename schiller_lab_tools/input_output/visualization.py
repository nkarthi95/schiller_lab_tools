#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyevtk.hl import imageToVTK
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


# In[2]:


def write_vti(path, filename, d):
    """
    Saves data as a .vti file in a specified directory.

    This function converts all NumPy arrays in the input dictionary `d` 
    to Fortran order and saves them as point data in a .vti file using the 
    `imageToVTK` function. The file is saved in the specified directory 
    with the provided filename.

    Parameters
    ----------
    path : str
        The directory path where the .vti file will be saved.
    filename : str
        The name of the .vti file to be created (without the `.vti` extension).
    d : dict
        A dictionary containing NumPy arrays with known key names. 
        Each array represents point data to be included in the .vti file.

    Returns
    -------
    int
        Returns 1 to indicate the file has been saved successfully.

    Notes
    -----
    - All NumPy arrays in the dictionary are converted to Fortran order before saving.
    - The .vti file is created using the `imageToVTK` function, with the dictionary 
      data included as point data.
    - The filename should not include the `.vti` extension as it is appended automatically.

    Examples
    --------
    >>> import numpy as np
    >>> d = {
    ...     "temperature": np.random.rand(10, 10, 10),
    ...     "pressure": np.random.rand(10, 10, 10)
    ... }
    >>> write_vti("/path/to/directory", "output_file", d)
    1
    """
    for i in d:
        d[i] = np.asfortranarray(d[i])
    
    imageToVTK(path, pointData=d)
    return 1


# In[3]:


def animate_plot(x, y, interval=50):
    """
    Create an animation of a 2D line plot over time.

    This function takes two lists of lists or numpy arrays representing the x and y 
    coordinates of data over multiple timesteps and generates an animated plot. The 
    animation can be rendered in a Jupyter notebook using the `HTML` object.

    Parameters
    ----------
    x : list of lists or numpy.ndarray
        The x-coordinates of the data, with shape (t, L), where t is the number 
        of timesteps and L is the length of data at each timestep.
    y : list of lists or numpy.ndarray
        The y-coordinates of the data, with shape (t, L), where t is the number 
        of timesteps and L is the length of data at each timestep.
    interval : int, optional
        The delay in milliseconds between frames in the animation. Default is 50.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        An animation object that can be rendered in a Jupyter notebook using 
        `IPython.display.HTML`.

    Examples
    --------
    >>> import numpy as np
    >>> from IPython.display import HTML
    >>> x = [np.linspace(0, 2 * np.pi, 100)] * 50
    >>> y = [np.sin(xi + i * 0.1) for i, xi in enumerate(x)]
    >>> ani = animate_plot(x, y, interval=100)
    >>> HTML(ani.to_jshtml())
    """
    fig, ax = plt.subplots()
    ln, = plt.plot(x[0], y[0])

    def get_min_max(arr):
        arr_min = 0
        arr_max = 0
        for i in range(len(arr)):
            if min(arr[i]) < arr_min:
                arr_min = min(arr[i])
            if max(arr[i]) > arr_max:
                arr_max = max(arr[i])
        return [arr_min, arr_max]

    def init():
        plt.ylim(get_min_max(y))
        plt.xlim(get_min_max(x))
        ln.set_data(x[0], y[0])
        return ln,

    def update(i):
        ln.set_data(x[i], y[i])
        return ln,

    ani = animation.FuncAnimation(fig, update, frames=len(x), init_func=init, interval=interval, blit=True)
    plt.close()
    return ani


# In[4]:


def animate_colormap(data, axs_labels=None, times=None, c_label=None, interval=50, sz=5, cm='bwr'):
    """
    Create an animation of a 2D colormap over time.

    This function takes a 3D numpy array representing time-dependent 2D data and 
    generates an animated colormap. The animation can be rendered in a Jupyter 
    notebook using the `HTML` object.

    Parameters
    ----------
    data : numpy.ndarray
        A 3D array with shape (t, L, M), where t is the number of timesteps, 
        and L and M represent the dimensions of each data slice (e.g., rows and columns).
    axs_labels : list of str, optional
        A list of length 2 containing strings for the x-axis and y-axis labels, 
        in the 0th and 1st positions respectively. Default is None.
    times : numpy.ndarray, optional
        A 1D array of length t, where each value represents the time corresponding 
        to each timestep. Default is None.
    c_label : str, optional
        A label for the colormap (color bar). Default is None.
    interval : int, optional
        The delay in milliseconds between frames in the animation. Default is 50.
    sz : int, optional
        The size of the plot figure. Default is 5.
    cm : str, optional
        The colormap to use for the animation. Default is 'bwr'.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        An animation object that can be rendered in a Jupyter notebook using 
        `IPython.display.HTML`.

    Examples
    --------
    >>> import numpy as np
    >>> from IPython.display import HTML
    >>> data = np.random.random((50, 100, 100))  # Example 3D array (50 timesteps, 100x100 grid)
    >>> ani = animate_colormap(data, axs_labels=["X-axis", "Y-axis"], times=np.arange(50))
    >>> HTML(ani.to_jshtml())
    """
    def init():
        img.set_data(data[0])
        vmin = np.amin(data[0])
        vmax = np.amax(data[0])
        img.set_clim(vmin, vmax)
        if times is not None:
            ax.set(title=f"Time = {times[0]}")
        return (img,)

    def update(i):
        img.set_data(data[i])
        vmin = np.amin(data[i])
        vmax = np.amax(data[i])
        img.set_clim(vmin, vmax)
        if times is not None:
            ax.set(title=f"Time = {times[i]}")
        return (img,)

    fig, ax = plt.subplots(1, 1, figsize=(sz, sz))
    img = ax.imshow(data[0], cmap=cm, vmin=np.amin(data[0]), vmax=np.amax(data[0]))
    if axs_labels is not None:
        ax.set_xlabel(axs_labels[0])
        ax.set_ylabel(axs_labels[1])
    fig.colorbar(img, ax=ax, orientation="horizontal", label=c_label, pad=0.2)
    ani = animation.FuncAnimation(fig, update, frames=len(data), init_func=init, interval=interval, blit=True)
    plt.close()
    return ani

