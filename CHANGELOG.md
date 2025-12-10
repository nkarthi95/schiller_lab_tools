# Changelog

## [0.1.2] - 2025-12-19

### Fixed
- Resolved bug where passing in `npart = 0` to `schiller_lab_tools.data.droplets.draw_ellipsoid_in_box` would raise an error by output `np.nan` for particle coordinates and orientations. 

## [0.1.1] - 2025-11-21

### Added
- Output nematic director from particle_properties.order.nematic in addition to nematic order parameter

### Fixed
- Resolved issue with particle_properties.order.nematic calling of core Freud function

## [0.1.0] - 2025-11-20

### Added
- Initial release of the package after refactor
