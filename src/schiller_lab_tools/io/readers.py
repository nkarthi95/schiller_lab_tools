import h5py
import numpy as np
import pandas as pd

def read_hdf5(filename):
    """
    Read the first dataset from an HDF5 file and return it as a NumPy array.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file to be read.

    Returns
    -------
    ndarray or int
        The first dataset in the file, returned as a NumPy array in
        Fortran order. Returns ``0`` if the file cannot be opened or
        an error occurs during reading.

    Notes
    -----
    The function reads only the first dataset found in the file.
    No reshaping or additional processing is performed; the dataset
    is returned in its native layout.

    Examples
    --------
    >>> data = read_hdf5("example.h5")
    >>> isinstance(data, np.ndarray)
    True
    """
    try:
        with h5py.File(filename, 'r') as file:
            OutArray = file.get(list(file.keys())[0])
            data = np.array(OutArray)
    except:
        data = 0
    return data

def read_asc(path, headers=None):
    """
    Read simulation data from an ASC file and return a DataFrame and timestep.

    Parameters
    ----------
    path : str
        File path of the ASC file to read.
    headers : list of str, optional
        Custom column headers. If omitted, default headers are assigned based
        on the number of columns in the file.

    Returns
    -------
    md_df : pandas.DataFrame
        DataFrame containing the parsed ASC file data. If the file is empty,
        a zero-filled DataFrame with default headers is returned.
    t : int
        Timestep extracted from the file name. The value is taken as the
        substring following the last occurrence of the character ``'t'``.

    Notes
    -----
    If no headers are provided, default headers are inferred from the
    number of columns:
    
    * 13 columns → ['x','y','z','v_x','v_y','v_z','o_x','o_y','o_z',
                    'w_x','w_y','w_z','p_id']
    * More than 13 columns → ['x','y','z','v_x','v_y','v_z','o_x','o_y','o_z',
                              'w_x','w_y','w_z','Fb_x','Fb_y','Fb_z',
                              't_x','t_y','t_z','p_id']
                            
    Empty ASC files trigger creation of a default DataFrame with these
    headers and zero-filled values.

    Examples
    --------
    >>> df, t = read_asc("data/asc_file_t100.asc")
    >>> df.head()
    >>> t
    100

    >>> custom_headers = ["a", "b", "c"]
    >>> df, t = read_asc("data/asc_file_t100.asc", headers=custom_headers)
    >>> df.head()
    """
    md_properties = np.loadtxt(path)
    md_properties = md_properties.T
    n = md_properties.shape
    t = int(path.split('/')[-1].split("_")[-1].split('-')[0].split('t')[-1])

    if headers is None:
        if n[0] == 0:
            headers = np.array(['x', 'y', 'z', 'v_x', 'v_y', 'v_z', 'o_x', 'o_y', 'o_z', 
                                'w_x', 'w_y', 'w_z', 'p_id'])
            d = {h: np.zeros(2) for h in headers}
            md_df = pd.DataFrame(d)
            return md_df, t
        elif n[0] == 13:
            headers = np.array(['x', 'y', 'z', 'v_x', 'v_y', 'v_z', 'o_x', 'o_y', 'o_z', 
                                'w_x', 'w_y', 'w_z', 'p_id'])
        elif n[0] > 13:
            headers = np.array(['x', 'y', 'z', 'v_x', 'v_y', 'v_z', 'o_x', 'o_y', 'o_z', 
                                'w_x', 'w_y', 'w_z', 'Fb_x', 'Fb_y', 'Fb_z', 't_x', 't_y', 
                                't_z', 'p_id'])

    d = {header: np.asanyarray(md_properties[i]) if isinstance(md_properties[i], np.ndarray) 
         else [md_properties[i]] for i, header in enumerate(headers)}

    md_df = pd.DataFrame(d)
    return md_df, t

def rewrite_asc_file(path):
    """
    Rewrite an ASC file to enforce NumPy-parsable numeric formatting.

    Parameters
    ----------
    path : str
        Absolute or relative path of the ASC file to be rewritten.

    Returns
    -------
    int
        Returns ``1`` after the file is successfully rewritten.

    Notes
    -----
    The function reads the ASC file line by line, inspects each numeric
    value, and rewrites entries that lack valid scientific notation
    formatting (e.g., values missing an ``'E'``). The corrected content
    is written back to the same file path, overwriting the original.
    This resolves formatting issues common in ASC files produced during
    checkpoint restarts or initial LB3D runs.

    Examples
    --------
    >>> rewrite_asc_file("output/asc_file.asc")
    1
    """
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    l = len(lines[0].split())

    for i in range(len(lines)):
        line = lines[i]
        line = line.split()
        for j in range(l - 1):
            curr = line[j]
            split_term = curr.split('E')
            if len(split_term) == 1:
                correct_term = split_term[0][:-4] + 'E' + split_term[0][-4:]
                line[j] = correct_term
        lines[i] = ' '.join(line)

    with open(path, 'w', encoding='utf-8') as r:
        for line in lines:
            r.write(line + '\n')

    return 1