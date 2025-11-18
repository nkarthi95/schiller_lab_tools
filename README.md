# schiller_lab_tools

`schiller_lab_tools` is a Python library for generating, analyzing, and visualizing microstructures and related fields. It provides tools for creating synthetic minimal surfaces, computing structure factors, extracting geometric descriptors, and integrating these analyses into scientific workflows. The library is built to support research in soft matter, complex fluids, porous materials, and multiphase systems.

## Features

* **Minimal surface generation**
  Generate synthetic gyroid structures and other minimal surface fields for testing and benchmarking.

* **Microstructure analysis**
  Compute structure factors (1D and projected), curvature fields, and geometric descriptors.

* **Field and geometry utilities**
  Access tools for manipulating 3D scalar and vector fields, interfaces, and derived quantities.

* **Visualization helpers**
  Plot and animate microstructure data using functions designed for quick inspection and documentation.

* **Sphinx-Gallery examples**
  Fully executable examples demonstrating usage, included in the documentation.

## Installation

If installing from a local clone in a fresh build, first setup the python environment

```
conda env create -f environment.yml
conda activate LB3D
```

Then install `schiller_lab_tools` from a local clone:

```
git clone https://github.com/nkarthi95/schiller_lab_tools.git
cd schiller_lab_tools
pip install .
```

<!-- ```
pip install schiller_lab_tools
``` -->

## Documentation

Full documentation, including API references and example galleries, is available at:

```
website_TBD
```

The documentation is built with Sphinx and Sphinx-Gallery. Example scripts live in:

```
docs/examples/
```

and are executed during the build process to produce reproducible figures.

## Usage Example

```python
import numpy as np
from schiller_lab_tools.data import minimal_surfaces
from schiller_lab_tools.microstructure import spherically_averaged_structure_factor

L = 128
field = minimal_surfaces.gyroid(L, L, L, reps=2)
k, S = scattering.spherically_averaged_structure_factor(field)
```

## Repository Structure

```
src/schiller_lab_tools/
    data/
    io/
    lb3d/
    microstructure/
    particle_properties/
    visualization/

docs/
    api/
    examples/
    conf.py
    index.rst
    Makefile
    make.bat
```

## Contributing

Pull requests are welcome. Ensure that new features include:

* Unit tests
* Documentation updates
* Example scripts when appropriate

## License

This project is licensed under the MIT License.