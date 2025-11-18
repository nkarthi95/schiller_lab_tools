"""
Projected Structure Factor of a Gyroid
======================================

Compute and visualize 2D projections of the structure factor.
"""


import numpy as np
import matplotlib.pyplot as plt
from schiller_lab_tools.data import minimal_surfaces
from schiller_lab_tools.microstructure import scattering

L = 32
test_gyroid = minimal_surfaces.gyroid(L, L, L, reps = 3)
_, S = scattering.structure_factor(test_gyroid)

fig, axs = plt.subplots(1, 3, figsize = (9, 3))

for i in range(len(axs)):
    ax = axs[i]
    S_projection = np.fft.fftshift(S).sum(axis = i)
    k = np.fft.fftshift(np.fft.fftfreq(L))
    k_horizontal, k_vertical = np.meshgrid(k, k)
    ax.pcolormesh(k_horizontal, k_vertical, S_projection, cmap = "plasma")

fig.tight_layout()
plt.plot()