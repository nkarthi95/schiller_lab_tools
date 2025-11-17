import xdrlib
import numpy as np

def write_checkparams_xdr(filename, params):
    """
    Write simulation parameters to an XDR-encoded binary file.

    Parameters
    ----------
    filename : str
        Path to the output file that will receive the XDR-encoded data. Includes the .xdr
        format in the filename
    params : dict
        Dictionary containing all required simulation parameters.  
        The function expects the keys listed in ``double_fields`` and
        ``int_fields``. Values corresponding to ``double_fields`` must be
        float-convertible; values corresponding to ``int_fields`` must be
        int-convertible.

    Returns
    -------
    None
        The function writes data to disk and returns nothing.

    Notes
    -----
    The function serializes parameters using ``xdrlib.Packer`` in the
    following order:

    * 42 double-precision values, packed in the order specified by
      ``double_fields``.
    * 29 integer values, packed in the order specified by ``int_fields``.

    This ordering is strict and must match the expected structure of any
    downstream code reading the resulting XDR file. Missing keys or
    type-mismatched entries in ``params`` will produce errors during packing.

    The encoded buffer is written verbatim to ``filename`` in binary mode.

    """

    packer = xdrlib.Packer()

    # === Write 42 doubles ===
    double_fields = [
        "fr", "fg", "fb", "beta",
        "amass_b", "amass_r", "amass_s",
        "tau_b", "tau_r", "tau_s", "tau_d",
        "taubulk_b", "taubulk_r", "taubulk_s", "taubulk_d",
        "g_br", "g_bs", "g_ss", "g_wr", "g_wb",
        "tau_wr", "tau_wb", "shear_u", "shear_omega",
        "g_accn", "g_accn_x", "g_accn_y",
        "s03_r", "s05_r", "s11_r", "s14_r", "s17_r",
        "s03_b", "s05_b", "s11_b", "s14_b", "s17_b",
        "s03_s", "s05_s", "s11_s", "s14_s", "s17_s"
    ]

    for field in double_fields:
        packer.pack_double(params[field])

    # === Write 29 integers ===
    int_fields = [
        "n_sci_int", "n_sci_sur", "n_sci_od", "n_sci_wd", "n_sci_dir",
        "n_sci_vel", "n_sci_flo", "n_sci_arrows", "n_sci_rock", "n_sci_rock_colour",
        "n_sci_rock_rho_r", "n_sci_rock_rho_b", "n_sci_pressure", "n_sci_fluxz",
        "n_sci_massfluxz", "n_sci_profile", "n_sci_profile_dump", "n_checkpoint",
        "g_accn_min", "g_accn_max", "g_accn_min_x", "g_accn_max_x",
        "g_accn_min_y", "g_accn_max_y"
    ]

    for field in int_fields:
        packer.pack_int(params[field])

    # === Write to file ===
    with open(filename, "wb") as f:
        f.write(packer.get_buffer())

def write_checkpoint_xdr(filename, domain):
    """
    Write a fluid-domain checkpoint file in XDR format using Fortran-compatible layout.

    Parameters
    ----------
    filename : str
        Path to the output XDR checkpoint file. Includes .xdr in format specifier.
    domain : [nx,ny,nz,indices] shaped ndarray
             Index 0-18 contains red fluid
             Index 19-38 contains blue fluid
             Index 39,40 contain rock state

    Returns
    -------
    None
        Writes binary XDR data to disk and returns nothing.

    Notes
    -----
    Fluid and rock fields are serialized for each lattice site in
    Fortran ordering (z-major, then y, then x).
    ``n_r`` and ``n_b`` fields are written as doubles; rock fields as integers.
    The structure must match the expectations of the LB3D Fortran reader.

    The caller is responsible for ensuring consistent field definitions
    within each element of ``domain[x][y][z]``.
    """
    packer = xdrlib.Packer()

    nx, ny, nz, indices = domain.shape
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                for i in range(indices):
                    packer.pack_double(domain[x,y,z,i])

    # Write to file
    with open(filename, "wb") as f:
        f.write(packer.get_buffer())

def write_checktopo_xdr(filename, topo_dict):
    """
    Write a topology checkpoint file (checktopo*.xdr) in XDR format.

    Parameters
    ----------
    filename : str
        Path to the output XDR file. Includes .xdr in the filename
    topo_dict : dict
        Dictionary containing topology metadata. Expected keys:
            * ``"cdims"`` : array_like of shape (3,)
              Domain decomposition dimensions.
            * ``"all_ccoords"`` : array_like of shape (3 * nprocs,)
              Flattened list of per-rank coordinates.
            * ``"shear_sum"`` : float or None
              Optional shear-sum value to encode.

    Returns
    -------
    None
        Writes XDR-encoded topology data to ``filename``.

    Notes
    -----
    The function encodes the entries of ``topo_dict`` in the order expected
    by the LB3D topology reader. All values are written using XDR integer
    or double-precision encodings as appropriate. No consistency checks are
    performed; the caller is responsible for supplying a valid structure.

    """

    packer = xdrlib.Packer()

    cdims = topo_dict['cdims']
    all_ccoords = topo_dict['all_ccoords']
    shear_sum = topo_dict['shear_sum']

    # === Write cdims (3 ints) ===
    for val in cdims:
        packer.pack_int(val)

    # === Write all_ccoords (3 * nprocs ints) ===
    for coord in all_ccoords:
        packer.pack_int(coord)

    # === Optional shear_sum ===
    if shear_sum is not None:
        packer.pack_double(shear_sum)

    # === Save to file ===
    with open(filename, "wb") as f:
        f.write(packer.get_buffer())

def write_md_checkpoint_xdr(
    filename,
    particles,
    *,
    use_rotation=False,
    polydispersity=False,
    magdispersity=False,
    steps_per_lbe_step=1,
    include_rhof=False,
    interaction=None,
    ladd_props=None,
    boundary=None,
    first_inserted_puid=None,
    n_spec=None
):
    """
    Write a particle checkpoint file (md-checkpoint*.xdr) in
    Fortran-compatible XDR format with full conditional field support.

    Parameters
    ----------
    filename : str
        Path to the output XDR file including the .xdr format.
    particles : list of dict
        Particle data records required by the MD configuration.
    use_rotation : bool, optional
        Include rotational quantities if True.
    polydispersity : bool, optional
        Include polydisperse radius fields if True.
    magdispersity : bool, optional
        Include magnetic-moment fields if True.
    steps_per_lbe_step : int, optional
        MD-to-LBE stepping ratio.
    include_rhof : bool, optional
        Include local-fluid density fields if True.
    interaction : str or None, optional
        If ``"ladd"``, LADD mass-exchange fields are expected.
    ladd_props : dict or None, optional
        LADD mass-exchange parameters.
    boundary : str or None, optional
        If ``"periodic_inflow"``, special metadata is included.
    first_inserted_puid : int or None, optional
        Required when ``boundary="periodic_inflow"``.
    n_spec : int or None, optional
        Number of LADD species.

    Returns
    -------
    None
        Writes XDR-encoded particle data.

    Notes
    -----
    The encoded structure depends on the active configuration flags.
    """
    packer = xdrlib.Packer()

    # Write total number of particles
    packer.pack_int(len(particles))

    for p in particles:
        # Position and velocity
        for val in p["x"]:
            packer.pack_double(val)
        for val in p["v"]:
            packer.pack_double(val)

        # Rotation fields
        if use_rotation:
            for val in p["q"]:
                packer.pack_double(val)
            for val in p["w"]:
                packer.pack_double(val)

        # ID and next-step velocity
        packer.pack_int(p["uid"])
        for val in p["vnew"]:
            packer.pack_double(val)

        if use_rotation:
            for val in p["qnew"]:
                packer.pack_double(val)
            for val in p["wnew"]:
                packer.pack_double(val)

        # Master flag
        packer.pack_int(p["master"])

        # Polydispersity
        if polydispersity:
            packer.pack_double(p["R_orth"])
            packer.pack_double(p["R_para"])

        # Magnetic moment
        if magdispersity:
            packer.pack_double(p["mag"])

        # Velocity accumulator or fallback
        if steps_per_lbe_step > 1:
            for val in p["v_fluid_acc"]:
                packer.pack_double(val)
            if use_rotation:
                for val in p["ws_fluid_acc"]:
                    packer.pack_double(val)
        else:
            for val in p["v"]:
                packer.pack_double(val)
            if use_rotation:
                for val in p["ws"]:
                    packer.pack_double(val)

        # Fluid interaction
        for val in p["v_fluid"]:
            packer.pack_double(val)
        if use_rotation:
            for val in p["ws_fluid"]:
                packer.pack_double(val)

        for val in p["f_fluid"]:
            packer.pack_double(val)
        if use_rotation:
            for val in p["t_fluid"]:
                packer.pack_double(val)

        # Optional fluid density field
        if include_rhof:
            packer.pack_double(p["rhof"])

    # LADD interaction extras
    if interaction == 'ladd':
        for val in [ladd_props["pfr"], ladd_props["pfb"], ladd_props["pfg"]]:
            packer.pack_double(val)
        for val in ladd_props['global_mass_change']:
            packer.pack_double(val)
        for val in ladd_props['global_mass_target']:
            packer.pack_double(val)

    # Inflow boundary condition
    if boundary == 'periodic_inflow':
        packer.pack_int(first_inserted_puid)

    # Write file
    with open(filename, "wb") as f:
        f.write(packer.get_buffer())

def read_checkparams_xdr(filename):
    """
    Read a checkparams*.xdr file and return all encoded simulation parameters.

    Parameters
    ----------
    filename : str
        Path to the XDR-encoded parameter file. .xdr included in filename

    Returns
    -------
    dict
        Dictionary containing all unpacked parameters.
        Includes 42 double-precision fields and 29 integer fields.

    Notes
    -----
    Values are decoded using xdrlib.Unpacker in fixed order.
    ``unpacker.done()`` ensures no remaining bytes.

    Examples
    --------
    >>> params = read_checkparams_xdr("checkparams.xdr")
    >>> params["beta"]
    0.1234
    """

    with open(filename, "rb") as f:
        data = f.read()

    unpacker = xdrlib.Unpacker(data)

    # === 1. Field names ===
    double_fields = [
        "fr", "fg", "fb", "beta", "amass_b", "amass_r", "amass_s",
        "tau_b", "tau_r", "tau_s", "tau_d",
        "taubulk_b", "taubulk_r", "taubulk_s", "taubulk_d",
        "g_br", "g_bs", "g_ss", "g_wr", "g_wb",
        "tau_wr", "tau_wb", "shear_u", "shear_omega",
        "g_accn", "g_accn_x", "g_accn_y",
        "s03_r", "s05_r", "s11_r", "s14_r", "s17_r",
        "s03_b", "s05_b", "s11_b", "s14_b", "s17_b",
        "s03_s", "s05_s", "s11_s", "s14_s", "s17_s"
    ]

    int_fields = [
        "n_sci_int", "n_sci_sur", "n_sci_od", "n_sci_wd", "n_sci_dir",
        "n_sci_vel", "n_sci_flo", "n_sci_arrows", "n_sci_rock", "n_sci_rock_colour",
        "n_sci_rock_rho_r", "n_sci_rock_rho_b", "n_sci_pressure", "n_sci_fluxz",
        "n_sci_massfluxz", "n_sci_profile", "n_sci_profile_dump",
        "n_checkpoint",
        "g_accn_min", "g_accn_max", "g_accn_min_x", "g_accn_max_x",
        "g_accn_min_y", "g_accn_max_y"
    ]

    # === 2. Read values ===
    doubles = {name: unpacker.unpack_double() for name in double_fields}
    ints = {name: unpacker.unpack_int() for name in int_fields}

    unpacker.done()  # optional: raises if not fully unpacked

    return {**doubles, **ints}

def read_checkpoint_xdr(filenames, nx, ny, nz, topology, Q=19):
    """
    Read one or more Fortran-compatible checkpoint*.xdr files and reconstruct
    the full fluid domain into a NumPy array.

    Parameters
    ----------
    filenames : list of str or str
        One or more XDR checkpoint files. Multiple files correspond to
        domain-decomposed (multicore) output.
    nx, ny, nz : int
        Dimensions of the reconstructed global fluid domain along x, y, and z.
    topology : dict
        Topology information describing how subdomains map into the global
        domain. Expected to contain decomposition metadata such as rank
        coordinates or layout needed to assemble the full 3D array.
    Q : int, optional
        Number of lattice velocity directions in the LB model. Default is 19.

    Returns
    -------
    ndarray of shape (nx, ny, nz, 2*Q + 4)
        Fully reconstructed checkpoint array. Channels are organized as:
            * indices 0 : Q       — red-fluid mass distributions
            * indices Q : 2Q      — blue-fluid mass distributions
            * indices 2Q         — rock-state field
            * indices 2Q+1 ...   — additional rock metadata (e.g.,
              rock colour and related fields)

    Notes
    -----
    The function reads each XDR file using Fortran-compatible ordering and
    merges all subdomain data according to the supplied ``topology`` mapping.
    The LB distributions for red and blue fluids each occupy ``Q`` channels.
    Rock-related fields follow after the fluid distributions. The caller must
    ensure that ``topology`` is consistent with the domain decomposition used
    when writing the checkpoint files.

    """

    topo_decomp = np.array(topology["cdims"])
    [x_decomp, y_decomp, z_decomp] = np.array(np.array([nx, ny, nz])/topo_decomp, dtype = int)

    topo_coords = np.array(topology["all_ccoords"])
    checkpt_files_count = topo_coords.size//3
    
    checkpoint_props = np.empty((nx, ny, nz, 2*Q+4))

    for i in range(checkpt_files_count):
        filename = filenames[i]
        slc = slice(i*3, (i+1)*3)
        [xmpi, ympi, zmpi]= topo_coords[slc]

        with open(filename, "rb") as f:
            data = f.read()

        upk = xdrlib.Unpacker(data)

        for x in range(x_decomp*xmpi, x_decomp*(xmpi+1)):
            for y in range(y_decomp*ympi, y_decomp*(ympi+1)):
                for z in range(z_decomp*zmpi, z_decomp*(zmpi+1)):
                    for box_prop in range(2*Q+4):
                        checkpoint_props[x,y,z,box_prop] = upk.unpack_double()

    return checkpoint_props

def read_checktopo_xdr(filename, nprocs=None):
    """
    Read a checktopo*.xdr topology file and return the domain-decomposition metadata.

    Parameters
    ----------
    filename : str
        Path to the XDR-encoded topology file which includes the xdr file format
    nprocs : int, optional
        Number of MPI ranks (subdomains). If not provided, the value is
        inferred from the remaining byte count in the file.

    Returns
    -------
    dict
        Dictionary containing:
            * ``"cdims"`` : list of int  
              Domain-decomposition dimensions along x, y, z (3 integers).
            * ``"all_ccoords"`` : list of int  
              Flattened list of rank coordinates of length ``3 * nprocs``.
            * ``"nprocs"`` : int  
              Number of ranks detected or provided.
            * ``"shear_sum"`` : float or None  
              Optional shear-sum value, present only if encoded in the file.

    Notes
    -----
    If ``nprocs`` is not supplied, it is inferred from the number of
    remaining bytes after the ``cdims`` block. Each rank contributes
    12 bytes (three 4-byte integers), and an additional 8 bytes may be
    present for a trailing ``shear_sum`` value.  
    The function uses ``xdrlib.Unpacker`` to decode the file in strict
    XDR order.

    Examples
    --------
    >>> topo = read_checktopo_xdr("checktopo.xdr")
    >>> topo["cdims"]
    [2, 2, 1]
    >>> topo["nprocs"]
    4
    """

    with open(filename, "rb") as f:
        data = f.read()

    upk = xdrlib.Unpacker(data)

    # === Read cdims (3 integers) ===
    cdims = [upk.unpack_int() for _ in range(3)]

    # === Determine nprocs if not provided ===
    if nprocs is None:
        remaining_bytes = len(data) - upk.get_position()
        # each int = 4 bytes, so 3 * nprocs ints = 12 * nprocs bytes
        nprocs = remaining_bytes // 12
        has_shear_sum = (remaining_bytes % 12) == 8
    else:
        has_shear_sum = (len(data) - upk.get_position() - 12 * nprocs) >= 8

    # === Read all_ccoords (3 * nprocs integers) ===
    all_ccoords = [upk.unpack_int() for _ in range(3 * nprocs)]

    # === Read optional shear_sum ===
    shear_sum = None
    if has_shear_sum:
        shear_sum = upk.unpack_double()

    return {
        "cdims": cdims,
        "all_ccoords": all_ccoords,
        "nprocs": nprocs,
        "shear_sum": shear_sum
    }

def read_md_checkpoint_xdr(
    filename,
    *,
    use_rotation=False,
    polydispersity=False,
    magdispersity=False,
    steps_per_lbe_step=1,
    include_rhof=False,
    interaction=None,
    n_spec=None,
    boundary=None
):
    """
    Read an LB3D particle checkpoint file (md-checkpoint*.xdr) and decode all
    particle and interaction data according to the active configuration flags.

    Parameters
    ----------
    filename : str
        Path to the XDR-encoded MD checkpoint file.
    use_rotation : bool, optional
        If True, decode rotational quantities (angular velocity, torque).
    polydispersity : bool, optional
        If True, decode per-particle radius fields (e.g., ``R_orth``, ``R_para``).
    magdispersity : bool, optional
        If True, decode magnetic-moment fields.
    steps_per_lbe_step : int, optional
        MD-to-LBE step ratio used to properly reconstruct velocity accumulator data.
    include_rhof : bool, optional
        If True, decode local surrounding-fluid density (``rhof``) fields.
    interaction : str or None, optional
        Interaction model identifier. If ``"ladd"``, additional LADD fields are read.
    n_spec : int or None, optional
        Number of species relevant for LADD mass exchange arrays. Required if
        ``interaction="ladd"``.
    boundary : str or None, optional
        Boundary-condition model. If ``"periodic_inflow"``, the checkpoint contains
        an additional ``first_inserted_puid`` key.

    Returns
    -------
    dict
        A dictionary with the following keys:
            * ``"particles"`` : list of dict  
              Each entry contains per-particle state fields, with included
              quantities determined by the configuration flags.
            * ``"ladd_data"`` : dict or None  
              Present only when ``interaction="ladd"``; contains
              ``pfr``, ``pfb``, ``pfg``, ``global_mass_change``,
              and ``global_mass_target``.
            * ``"first_inserted_puid"`` : int or None  
              Present only when ``boundary="periodic_inflow"``.

    Notes
    -----
    This reader expects the XDR layout produced by ``write_md_checkpoint_xdr``.
    The presence or absence of fields is controlled strictly by the provided
    keyword flags. Inconsistent combinations of flags and file content will
    cause unpacking errors.

    """

    particles = []
    ladd_data = {}
    first_inserted_puid = None

    with open(filename, "rb") as f:
        data = f.read()

    upk = xdrlib.Unpacker(data)
    n_global = upk.unpack_int()

    for _ in range(n_global):
        p = {}
        p["x"] = [upk.unpack_double() for _ in range(3)]
        p["v"] = [upk.unpack_double() for _ in range(3)]

        if use_rotation:
            p["q"] = [upk.unpack_double() for _ in range(4)]
            p["w"] = [upk.unpack_double() for _ in range(3)]

        p["uid"] = upk.unpack_int()
        p["vnew"] = [upk.unpack_double() for _ in range(3)]

        if use_rotation:
            p["qnew"] = [upk.unpack_double() for _ in range(4)]
            p["wnew"] = [upk.unpack_double() for _ in range(3)]

        p["master"] = upk.unpack_int()

        if polydispersity:
            p["R_orth"] = upk.unpack_double()
            p["R_para"] = upk.unpack_double()

        if magdispersity:
            p["mag"] = upk.unpack_double()

        if steps_per_lbe_step > 1:
            p["v_fluid_acc"] = [upk.unpack_double() for _ in range(3)]
            if use_rotation:
                p["ws_fluid_acc"] = [upk.unpack_double() for _ in range(3)]
        else:
            p["v"] = [upk.unpack_double() for _ in range(3)]
            if use_rotation:
                p["ws"] = [upk.unpack_double() for _ in range(3)]

        p["v_fluid"] = [upk.unpack_double() for _ in range(3)]
        if use_rotation:
            p["ws_fluid"] = [upk.unpack_double() for _ in range(3)]

        p["f_fluid"] = [upk.unpack_double() for _ in range(3)]
        if use_rotation:
            p["t_fluid"] = [upk.unpack_double() for _ in range(3)]

        if include_rhof:
            p["rhof"] = upk.unpack_double()

        particles.append(p)

    if interaction == 'ladd':
        if n_spec is None:
            raise ValueError("n_spec must be provided when interaction == 'ladd'")
        ladd_data["pfr"] = upk.unpack_double()
        ladd_data["pfb"] = upk.unpack_double()
        ladd_data["pfg"] = upk.unpack_double()
        ladd_data["global_mass_change"] = [upk.unpack_double() for _ in range(n_spec)]
        ladd_data["global_mass_target"] = [upk.unpack_double() for _ in range(n_spec)]

    if boundary == 'periodic_inflow':
        first_inserted_puid = upk.unpack_int()

    # print("Remaining bytes:", len(data) - upk.get_position())
    
    return {
        "particles": particles,
        "ladd_data": ladd_data if ladd_data else None,
        "first_inserted_puid": first_inserted_puid,
    }