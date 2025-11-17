import xdrlib
import numpy as np

def write_checkparams_xdr(filename, params):
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

def write_checkpoint_xdr(filename, domain, nx, ny, nz):
    """
    Write a checkpoint*.xdr file for the fluid domain (Fortran-compatible layout).
    
    Args:
        filename (str): Output file path.
        domain (3D list of dict): domain[x][y][z], each with:
            - 'n_r': list of floats
            - optionally: 'n_b': list of floats (if not SINGLEFLUID)
            - 'rock_state', 'rock_colour', 'rock_colour_r', 'rock_colour_b'
        cdims (list of 3 ints): Optional domain decomposition dimensions.
        shear_sum (float): Optional shear_sum value.
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
    Write checktopo*.xdr file.
    
    Args:
        filename (str): Output path.
        cdims (list of int): 3-element list of domain decomposition dimensions.
        all_ccoords (list of int): Flattened list of all rank coordinates (3 * nprocs).
        shear_sum (float or None): Optional shear_sum value.
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
    interaction=None,  # e.g., 'ladd'
    ladd_props = None,
    boundary=None,  # e.g., 'periodic_inflow'
    first_inserted_puid=None,
    n_spec=None
):
    """
    Write a Fortran-compatible md-checkpoint*.xdr file with full conditional support.
    
    Args:
        filename (str): Output file name.
        particles (list of dict): Particle list with required fields.
        use_rotation (bool): Include rotation fields.
        polydispersity (bool): Include R_orth, R_para.
        magdispersity (bool): Include mag field.
        steps_per_lbe_step (int): Controls velocity accumulator.
        include_rhof (bool): Include rhof if LADD_SURR_RHOF is active.
        interaction (str): 'ladd' or None.
        global_mass_change (list): Only if interaction == 'ladd'.
        global_mass_target (list): Only if interaction == 'ladd'.
        boundary (str): 'periodic_inflow' or None.
        first_inserted_puid (int): Required if boundary == 'periodic_inflow'.
        n_spec (int): Number of species for ladd mass arrays.
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
    Read a Fortran-compatible checkpoint*.xdr file and return a numpy array of the checkpoint data and rock files.
    Index 0:Q stores the mass distribution data for the red fluid
    Index Q:2Q stores the mass distribution data for the blue fluid
    Index 2Q: stores the data for the rock state
    Supports multicore output files
    
    Args:
        filename (str): Path to .xdr checkpoint file.
        nx, ny, nz (int): Dimensions of the fluid subdomain (x, y, z).
        topology (dict): Instructs the loop how to perform domain reconstruction
    
    Returns:
        checkpoint_props: (nx, ny, nz, 2*Q + 4)
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
    Read a Fortran-compatible md-checkpoint*.xdr file with full conditional support.
    
    Returns:
        dict with keys:
            - particles: list of particle dicts
            - ladd_data: dict with keys pfr, pfb, pfg, global_mass_change, global_mass_target
            - first_inserted_puid: int (if periodic_inflow)
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