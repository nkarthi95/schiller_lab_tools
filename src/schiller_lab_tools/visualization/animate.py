#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def animate_plot(x, y, interval=50):
    """
    Create an animated 2D line plot over multiple timesteps.

    Parameters
    ----------
    x : list of lists or ndarray of shape (T, L)
        X-coordinates for each timestep. ``T`` is the number of frames and
        ``L`` is the number of points per frame.
    y : list of lists or ndarray of shape (T, L)
        Y-coordinates for each timestep, matching the structure of ``x``.
    interval : int, optional
        Delay between frames in milliseconds. Default is 50.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Animation object suitable for display in Jupyter notebooks (e.g.,
        via ``IPython.display.HTML``).

    Examples
    --------
    >>> x = [np.linspace(0, 2*np.pi, 100)] * 50
    >>> y = [np.sin(xi + i*0.1) for i, xi in enumerate(x)]
    >>> ani = animate_plot(x, y, interval=100)
    >>> from IPython.display import HTML
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


def animate_colormap(data, axs_labels=None, times=None, c_label=None,
                     interval=50, sz=5, cm="bwr"):
    """
    Create an animated colormap from time-dependent 2D data.

    Parameters
    ----------
    data : ndarray of shape (T, L, M)
        Time-indexed 2D fields. ``T`` is the number of timesteps, and
        ``L`` and ``M`` are the spatial dimensions.
    axs_labels : list of str, optional
        Axis labels ``[xlabel, ylabel]``. Default is None.
    times : ndarray of shape (T,), optional
        Time values associated with each frame. Default is None.
    c_label : str, optional
        Label for the colorbar. Default is None.
    interval : int, optional
        Delay between frames in milliseconds. Default is 50.
    sz : int, optional
        Figure size in inches. Default is 5.
    cm : str, optional
        Matplotlib colormap name. Default is "bwr".

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Animation object suitable for display in Jupyter notebooks
        (e.g., via ``IPython.display.HTML``).

    Examples
    --------
    >>> data = np.random.random((50, 100, 100))
    >>> ani = animate_colormap(data, axs_labels=["X-axis", "Y-axis"],
    ...                        times=np.arange(50))
    >>> from IPython.display import HTML
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

