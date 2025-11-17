import numpy as np


def pressure_jump(pressure):
    """
    Calculate the pressure difference between the interior and exterior of a droplet.

    This function calculates the difference in scalar pressure between the center of the droplet 
    (assumed to be at the center of the pressure field) and the exterior (assumed to be at the 
    corner of the pressure field).

    :param pressure: 
        A 3D numpy array representing the scalar pressure field of the system. It is assumed that 
        the pressure at the center of the droplet is located at the center of the array, and the 
        pressure at the exterior is at a corner of the array.
    :type pressure: numpy.ndarray

    :return: 
        The difference in pressure between the interior (center) and the exterior (corner) of 
        the droplet.
    :rtype: float

    :note: 
        - The function assumes that the pressure field is centered around the droplet, with the 
          exterior pressure defined at the corner of the field.
        - The pressure difference is calculated as the scalar pressure difference between the center 
          and the corner.

    :example:
        >>> pressure = np.random.random((10, 10, 10))  # Example pressure field
        >>> pressure_jump(pressure)
        0.025  # Example output for pressure difference
    """
    center = tuple([l // 2 for l in pressure.shape])
    dP = pressure[center] - pressure[0, 0, 0]
    return dP