"""
1D Structure Factor of a Gyroid
===============================

Compute and visualize the 1D structure factor of a gyroid.
"""


import numpy as np
import matplotlib.pyplot as plt

from schiller_lab_tools.data import minimal_surfaces
from schiller_lab_tools.microstructure import scattering

L = 128
reps = np.arange(1, 5)

fig, ax = plt.subplots(figsize=(4, 3))

for rep in reps:
    field = minimal_surfaces.gyroid(L, L, L, reps=rep)
    k, S = scattering.spherically_averaged_structure_factor(field)
    ax.plot(k, S, label=f"reps = {rep}")

ax.set(xlabel = r"$k$", ylabel = r"$I_k$", xlim = [0, 0.5])
ax.legend()
fig.tight_layout()
plt.plot()