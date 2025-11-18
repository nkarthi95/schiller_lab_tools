import numpy as np
from pyevtk.hl import imageToVTK

def write_vti(path, filename, d):
    """
    Save a set of NumPy arrays as point data in a .vti file.

    Parameters
    ----------
    path : str
        Directory where the .vti file will be written.
    filename : str
        Base filename (without the ``.vti`` extension).
    d : dict
        Dictionary mapping string keys to NumPy arrays. Each array is
        treated as point data and converted to Fortran order before saving.

    Returns
    -------
    int
        Returns ``1`` when the file is successfully written.

    Notes
    -----
    All arrays in ``d`` are converted to Fortran order prior to export.
    The output file is generated using ``imageToVTK`` with the dictionary
    entries stored as VTK point data. The ``.vti`` extension is appended
    automatically; it must not be included in ``filename``.

    Examples
    --------
    >>> d = {"temperature": np.random.rand(10, 10, 10),
    ...      "pressure": np.random.rand(10, 10, 10)}
    >>> write_vti("/path/to/dir", "output_file", d)
    1
    """
    for i in d:
        d[i] = np.asfortranarray(d[i])
    
    imageToVTK(path + "/" + filename, pointData=d)
    return 1

def convert_xyz(input_df, core=['z', 'y', 'x', 'oz', 'oy', 'ox'], extend_col=None):
    """
    Convert a pandas DataFrame into an XYZ-formatted string.

    Parameters
    ----------
    input_df : pandas.DataFrame
        DataFrame containing atomic or particle data. Each row corresponds
        to one atom.
    core : list of str, optional
        Base list of column names that defines the output ordering.
        Defaults to ['z', 'y', 'x', 'oz', 'oy', 'ox'].
    extend_col : str or list of str or ndarray, optional
        Extra column(s) to append to ``core``. A single string is appended
        directly; a list or NumPy array contributes all elements.

    Returns
    -------
    str
        XYZ-formatted string. The first line contains the row count.
        Each subsequent line begins with the fixed symbol ``Ni`` followed
        by the ordered column values.

    Notes
    -----
    ``core`` is modified in place when ``extend_col`` is supplied. Pass a
    copy if the original ordering must be preserved. The output uses a
    hardcoded atomic symbol (``Ni``) for all rows.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "x": [0.0, 1.0],
    ...     "y": [0.1, 1.1],
    ...     "z": [0.2, 1.2],
    ...     "extra": [42, 43],
    ... })
    >>> print(convert_xyz(df, core=["z","y","x"], extend_col="extra"))
    2
    Ni 0.2 0.1 0.0 42 
    Ni 1.2 1.1 1.0 43 
    """
    out = ""
    out += str(input_df.shape[0]) + "\n"

    if isinstance(extend_col, str):
        core.append(extend_col)
    elif isinstance(extend_col, list):
        core += extend_col
    elif isinstance(extend_col, np.ndarray):
        core += extend_col.tolist()

    for _, row in input_df.iterrows():
        curr_row_str = "Ni "
        vars = row[core]
        for var in vars:
            curr_row_str += str(var) + " "
        curr_row_str += "\n"
        out += curr_row_str

    return out