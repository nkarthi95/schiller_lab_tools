import numpy as np

def fibonacci_sphere(N, R):
    # shift to center if needed by adding a center vector afterward
    i = np.arange(N)
    phi = np.pi * (3. - np.sqrt(5.))         # golden angle
    y = 1 - (2*i + 1)/N                      # y = cos(theta)
    r = np.sqrt(1 - y*y)
    theta = i * phi

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = y

    coords = np.column_stack((x,y,z))

    return np.multiply(coords, R)