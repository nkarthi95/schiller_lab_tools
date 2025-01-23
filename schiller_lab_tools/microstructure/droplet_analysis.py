#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def droplet_radius(density, Vp=0, np_sphere=0, rho_sphere=1):
    """
    Calculate the radius of a droplet from the density field and particle parameters.

    This function calculates the radius of a droplet based on the density distribution of two 
    density fields. It also considers particle volume, number of particles, and the density of 
    the particles in the calculation. The radius is computed using the mass and density difference 
    within the system.

    Parameters
    ----------
    density : numpy.ndarray
        A 3D numpy array representing the density field of the system, where each value encodes 
        the local density at a given point in the simulation box.

    Vp : float, optional, default=0
        The volume of a single particle. This parameter is used to estimate the mass of the droplet 
        based on the number of particles in the system.

    np_sphere : int, optional, default=0
        The total number of particles in the droplet or the system. This is used to calculate the 
        mass contribution of the particles in the droplet.

    rho_sphere : float, optional, default=1
        The density of the particles. This is used to calculate the total mass of the particles 
        contributing to the droplet.

    Returns
    -------
    float
        The radius of the droplet, calculated based on the mass and the density difference 
        between the droplet and the surrounding medium.

    Notes
    -----
    - The function assumes that the density field is centered on the droplet, and it calculates 
      the radius by considering the mass of the droplet and its density contrast with the surrounding 
      medium.
    - If `density` is an integer, the function returns `NaN`, as this indicates invalid input.

    Examples
    --------
    >>> density = np.random.random((10, 10, 10))  # Example density field
    >>> droplet_radius(density, Vp=1, np_sphere=100, rho_sphere=1.5)
    1.25  # Example output for droplet radius
    """
    if isinstance(density, int):
        return np.nan
    else:
        center = tuple([l // 2 for l in density.shape])
        rho_d = density[center]
        rho_m = density[0, 0, 0]
        mass = np.sum(density - rho_m) + 0.5 * Vp * np_sphere * rho_sphere
        R = (3. / 4. / np.pi * mass / (rho_d - rho_m)) ** (1. / 3.)
        return R


# In[3]:


def pressure_jump(pressure):
    """
    Calculate the pressure difference between the interior and exterior of a droplet.

    This function calculates the difference in scalar pressure between the center of the droplet 
    (assumed to be at the center of the pressure field) and the exterior (assumed to be at the 
    corner of the pressure field).

    Parameters
    ----------
    pressure : numpy.ndarray
        A 3D numpy array representing the scalar pressure field of the system. It is assumed that 
        the pressure at the center of the droplet is located at the center of the array, and the 
        pressure at the exterior is at a corner of the array.

    Returns
    -------
    float
        The difference in pressure between the interior (center) and the exterior (corner) of 
        the droplet.

    Notes
    -----
    - The function assumes that the pressure field is centered around the droplet, with the 
      exterior pressure defined at the corner of the field.
    - The pressure difference is calculated as the scalar pressure difference between the center 
      and the corner.

    Examples
    --------
    >>> pressure = np.random.random((10, 10, 10))  # Example pressure field
    >>> pressure_jump(pressure)
    0.025  # Example output for pressure difference
    """
    center = tuple([l // 2 for l in pressure.shape])
    dP = pressure[center] - pressure[0, 0, 0]
    return dP


# In[4]:


def inertia_tensor(cm, OutArray):
    """
    Calculate the inertia tensor of a 3D array with respect to its center of mass.

    This function computes the inertia tensor of a 3D numpy array `OutArray` based on the specified 
    center of mass `cm`. The inertia tensor is a measure of the distribution of mass and geometry 
    about the center of mass.

    Parameters
    ----------
    cm : array-like
        A 1D array or list of size 3 representing the center of mass of the array.
    OutArray : numpy.ndarray
        A 3D numpy array where each element represents a scalar mass density at that position.

    Returns
    -------
    numpy.ndarray
        A (3, 3) numpy matrix representing the inertia tensor of `OutArray`.

    Notes
    -----
    - The function assumes that `cm` specifies the center of mass in the coordinate system of 
      `OutArray`.
    - The inertia tensor is calculated using the outer product of position vectors shifted by 
      the center of mass, with mass weighting from the values of `OutArray`.

    Examples
    --------
    >>> cm = np.array([5.0, 5.0, 5.0])  # Center of mass
    >>> OutArray = np.random.random((10, 10, 10))  # Example density field
    >>> I = inertia_tensor(cm, OutArray)
    >>> print(I)
    [[83.5, 0.0, 0.0],
     [0.0, 84.2, 0.0],
     [0.0, 0.0, 82.8]]  # Example output inertia tensor
    """
    ind = np.transpose(np.indices(OutArray.shape), axes=(1, 2, 3, 0))
    pos = ind - cm
    r2 = np.einsum('ijkl,ijkl->ijk', pos, pos)          # inner product
    rr = np.einsum('ijkm,ijkn->ijkmn', pos, pos)        # outer product
    r2 = np.einsum('ijk,mn->ijkmn', r2, np.identity(3)) # multiply with unit matrix
    I = np.einsum('ijk,ijkmn->mn', OutArray, r2 - rr)   # sum m*(r2-rr)
    return I


# In[5]:


def gyration_tensor(cm, OutArray):
    """
    Calculate the gyration tensor of a 3D array with respect to its center of mass.

    This function computes the gyration tensor of a 3D numpy array `OutArray` using the provided 
    center of mass `cm`. The gyration tensor provides a measure of the spatial distribution of mass 
    around the center of mass.

    Parameters
    ----------
    cm : array-like
        A 1D array or list of size 3 representing the center of mass of the array.
    OutArray : numpy.ndarray
        A 3D numpy array where each element represents a scalar mass density at that position.

    Returns
    -------
    numpy.ndarray
        A (3, 3) numpy matrix representing the gyration tensor of `OutArray`.

    Notes
    -----
    - The gyration tensor is normalized by the total mass (sum of `OutArray`) and is calculated 
      using the second moment of the position vectors relative to the center of mass.
    - This tensor is useful for characterizing the shape and size of spatial distributions.

    Examples
    --------
    >>> cm = np.array([5.0, 5.0, 5.0])  # Center of mass
    >>> OutArray = np.random.random((10, 10, 10))  # Example density field
    >>> S = gyration_tensor(cm, OutArray)
    >>> print(S)
    [[0.33, 0.01, 0.02],
     [0.01, 0.35, 0.03],
     [0.02, 0.03, 0.37]]  # Example output gyration tensor
    """
    ind = np.transpose(np.indices(OutArray.shape), axes=(1, 2, 3, 0))
    pos = ind - cm
    rr = np.einsum('...m,...n->...mn', pos, pos)
    S = np.einsum('ijk,ijk...', OutArray, rr) / np.sum(OutArray)
    return S


# In[6]:


def deformation1(values1, m):
    """
    Calculate the deformation parameter of a droplet based on the eigenvalues of its inertia tensor.

    This function computes the principal axes' length scales of a droplet using the eigenvalues 
    of its inertia tensor and the total mass of the droplet. The deformation parameter `D` is 
    then calculated as a measure of the droplet's shape anisotropy.

    Parameters
    ----------
    values1 : numpy.ndarray
        A 1D numpy array containing the eigenvalues of the inertia tensor matrix, ordered by size.
    m : float
        The total mass of the droplet.

    Returns
    -------
    float
        The deformation parameter `D`, calculated as `(L - B) / (L + B)`, where `L` and `B` 
        are the largest and smallest principal axes of the droplet, respectively.

    Notes
    -----
    - The principal axes' lengths (`a`, `b`, `c`) are derived from the eigenvalues of the inertia tensor 
      and the droplet's total mass using the relationship:
      `a^2 = (5 / (2 * m)) * (λ2 + λ3 - λ1)`
    - This function assumes that the eigenvalues `values1` are sorted in descending order.

    Examples
    --------
    >>> values1 = np.array([10.0, 8.0, 6.0])  # Example eigenvalues
    >>> m = 5.0  # Total mass of the droplet
    >>> D = deformation1(values1, m)
    >>> print(D)
    0.3333333333333333  # Example output
    """
    a_squared = (5 / (2 * m)) * (values1[1] + values1[2] - values1[0])
    a = a_squared ** (1 / 2)
    b_squared = (5 / (2 * m)) * (values1[0] + values1[2] - values1[1])
    b = b_squared ** (1 / 2)
    c_squared = (5 / (2 * m)) * (values1[0] + values1[1] - values1[2])
    c = c_squared ** (1 / 2)
    L = max(a, b, c)
    B = min(a, b, c)
    D = (L - B) / (L + B)
    return D


# In[7]:


def inclination_angle(vectors, values1, m):
    """
    Calculate the inclination angle of a droplet with respect to its principal axis.

    This function computes the inclination angle in degrees between the droplet's 
    longest principal axis and a reference direction, based on the eigenvectors 
    and eigenvalues of its inertia tensor, as well as the droplet's mass.

    Parameters
    ----------
    vectors : numpy.ndarray
        A (3, 3) numpy array where each row corresponds to an eigenvector of the inertia tensor.
    values1 : numpy.ndarray
        A 1D numpy array of size 3 containing the eigenvalues of the inertia tensor.
    m : float
        The mass of the droplet.

    Returns
    -------
    float
        The inclination angle of the droplet with respect to its principal axis, in degrees.

    Notes
    -----
    - The principal axes' lengths (`a`, `b`, `c`) are derived from the eigenvalues of the inertia tensor 
      and the droplet's total mass using the relationship:
      `a^2 = (5 / (2 * m)) * (λ2 + λ3 - λ1)`
    - The inclination angle is computed as `atan(eigenvector[0] / eigenvector[2])`, where the eigenvector 
      corresponding to the largest principal axis is used.

    Examples
    --------
    >>> vectors = np.array([[0.577, 0.577, 0.577], 
    ...                     [0.707, -0.707, 0.0], 
    ...                     [0.408, 0.408, -0.816]])  # Example eigenvectors
    >>> values1 = np.array([10.0, 8.0, 6.0])  # Example eigenvalues
    >>> m = 5.0  # Total mass of the droplet
    >>> angle = inclination_angle(vectors, values1, m)
    >>> print(angle)
    30.0  # Example output
    """
    abc = []
    a_squared = (5 / (2 * m)) * (values1[1] + values1[2] - values1[0])
    a = a_squared ** (1 / 2)
    abc.append(a)
    b_squared = (5 / (2 * m)) * (values1[0] + values1[2] - values1[1])
    b = b_squared ** (1 / 2)
    abc.append(b)
    c_squared = (5 / (2 * m)) * (values1[0] + values1[1] - values1[2])
    c = c_squared ** (1 / 2)
    abc.append(c)
    maxpos = abc.index(max(abc))
    eigenvector = vectors[maxpos]
    inclination_angle = np.atan(eigenvector[0] / eigenvector[2])
    pi = np.pi
    inclination_angle = inclination_angle * (180 / pi)
    return inclination_angle

