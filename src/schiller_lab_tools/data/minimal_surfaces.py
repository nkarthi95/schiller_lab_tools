import numpy as np

# --------------------------------------------------------------------------
# Gyroids generation adapted from https://doi.org/10.1016/j.addma.2020.101548
# --------------------------------------------------------------------------

def gyroid(nx = 32, ny = 32, nz = 32, t = 0.5, reps = 1, n = 1):
    """
    Generate a scalar gyroid field on a 3D Cartesian grid.

    Parameters
    ----------
    nx, ny, nz : int, optional
        Grid dimensions along the x, y, and z axes. Default is 32 for each.
    t : float, optional
        Threshold level for the gyroid isosurface. The returned field is
        ``G**n - t**n``. Default is 0.5.
    reps : int or float, optional
        Spatial repetition factor used in the internal sine/cosine scaling
        functions. Controls unit cell size. Default is 1.
    n : int, optional
        Exponent applied to the base gyroid field before thresholding.
        Used to sharpen or smooth the interface. Default is 1.

    Returns
    -------
    field : ndarray of shape (nx, ny, nz)
        Scalar gyroid field defined as
        ``(Sx*Cy + Sy*Cz + Sz*Cx)**n - t**n``.
        Zero-level set approximates the classical gyroid minimal surface.

    Notes
    -----
    The gyroid field is evaluated on integer meshgrid coordinates ``X, Y, Z`` 
    generated with no indexing shift.
    """
    X, Y, Z = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1), np.arange(0, nz, 1))

    Sx = np.sin(2*np.pi*reps*X/nx)
    Sy = np.sin(2*np.pi*reps*Y/ny)
    Sz = np.sin(2*np.pi*reps*Z/nz)

    Cx = np.cos(2*np.pi*reps*X/nx)
    Cy = np.cos(2*np.pi*reps*Y/ny)
    Cz = np.cos(2*np.pi*reps*Z/nz)

    return (Sx*Cy + Sy*Cz + Sz*Cx)**n - t**n

def honeycomb(nx = 32, ny = 32, nz = 32, t = 0.5, reps = 1, axis = 'z'):
    """
    Generate a scalar honeycomb-like field on a 3D Cartesian grid.

    Parameters
    ----------
    nx, ny, nz : int, optional
        Grid dimensions along the x, y, and z axes. Default is 32 for each.
    t : float, optional
        Threshold parameter applied as ``prod - t**2``. Controls the level
        at which the honeycomb surface is cut. Default is 0.5.
    reps : int or float, optional
        Spatial repetition factor used by the internal functions ``_S`` and
        ``_C``. Governs periodicity and apparent unit-cell size. Default is 1.
    axis : {'x','y','z'}, optional
        Orientation selector controlling which analytic form of the
        honeycomb pattern is used:
        * 'z' → ``(Sx*Cy + Sy + Cx)**2 - t**2``
        * 'y' → ``(Sx + Cz + Sz*Cx)**2 - t**2``
        * other → ``(Cy + Sy*Cz + Sz)**2 - t**2``  
        Default is 'z'.

    Returns
    -------
    field : ndarray of shape (nx, ny, nz)
        Scalar honeycomb field. The zero level set of this field yields a
        hexagonal-lattice-like structure aligned with the chosen axis.

    Notes
    -----
    The field is evaluated over integer meshgrid
    coordinates generated without indexing shifts.
    """
    X, Y, Z = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1), np.arange(0, nz, 1))

    Sx = np.sin(2*np.pi*reps*X/nx)
    Sy = np.sin(2*np.pi*reps*Y/ny)
    Sz = np.sin(2*np.pi*reps*Z/nz)

    Cx = np.cos(2*np.pi*reps*X/nx)
    Cy = np.cos(2*np.pi*reps*Y/ny)
    Cz = np.cos(2*np.pi*reps*Z/nz)

    if axis == 'z':
        prod = (Sx*Cy + Sy + Cx)**2 - t**2
    elif axis == 'y':
        prod = (Sx + Cz + Sz*Cx)**2 - t**2
    else:
        prod = (Cy + Sy*Cz + Sz)**2 - t**2
    
    return prod