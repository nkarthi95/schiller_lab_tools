import numpy as np

# --------------------------------------------------------------------------
# Gyroids generation adapted from https://doi.org/10.1016/j.addma.2020.101548
# --------------------------------------------------------------------------

def _S(n, L, arr):
    return np.sin(2*np.pi*n*arr/L)

def _C(n, L, arr):
    return np.cos(2*np.pi*n*arr/L)

def gyroid(nx = 32, ny = 32, nz = 32, t = 0.5, reps = 1, n = 1):
    X, Y, Z = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1), np.arange(0, nz, 1))

    Sx = _S(reps, nx, X)
    Sy = _S(reps, ny, Y)
    Sz = _S(reps, nz, Z)

    Cx = _C(reps, nx, X)
    Cy = _C(reps, ny, Y)
    Cz = _C(reps, nz, Z)

    return (Sx*Cy + Sy*Cz + Sz*Cx)**n - t**n

def honeycomb(nx = 32, ny = 32, nz = 32, t = 0.5, reps = 1, axis = 'z'):
    X, Y, Z = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1), np.arange(0, nz, 1))

    Sx = _S(reps, nx, X)
    Sy = _S(reps, ny, Y)
    Sz = _S(reps, nz, Z)

    Cx = _C(reps, nx, X)
    Cy = _C(reps, ny, Y)
    Cz = _C(reps, nz, Z)

    if axis == 'z':
        prod = (Sx*Cy + Sy + Cx)**2 - t**2
    elif axis == 'y':
        prod = (Sx + Cz + Sz*Cx)**2 - t**2
    else:
        prod = (Cy + Sy*Cz + Sz)**2 - t**2
    
    return prod