import numpy as np
from pyevtk.hl import imageToVTK

def write_vti(path, filename, d):
    """
    Save data as a .vti file in a specified directory.

    This function converts all NumPy arrays in the input dictionary `d` 
    to Fortran order and saves them as point data in a .vti file using the 
    `imageToVTK` function. The file is saved in the specified directory 
    with the provided filename.

    :param path: 
        The directory path where the .vti file will be saved.
    :type path: str

    :param filename: 
        The name of the .vti file to be created (without the `.vti` extension).
    :type filename: str

    :param d: 
        A dictionary containing NumPy arrays with known key names. 
        Each array represents point data to be included in the .vti file.
    :type d: dict

    :return: 
        Returns 1 to indicate the file has been saved successfully.
    :rtype: int

    :note: 
        - All NumPy arrays in the dictionary are converted to Fortran order before saving.
        - The .vti file is created using the `imageToVTK` function, with the dictionary 
          data included as point data.
        - The filename should not include the `.vti` extension as it is appended automatically.

    :example:
        >>> import numpy as np
        >>> d = {
        ...     "temperature": np.random.rand(10, 10, 10),
        ...     "pressure": np.random.rand(10, 10, 10)
        ... }
        >>> write_vti("/path/to/directory", "output_file", d)
        1
    """
    for i in d:
        d[i] = np.asfortranarray(d[i])
    
    imageToVTK(path + "/" + filename, pointData=d)
    return 1

def convert_xyz(input_df, core=['z', 'y', 'x', 'oz', 'oy', 'ox'], extend_col=None):
    """
    Converts a pandas DataFrame into a string formatted according to the ``.xyz`` file convention.

    The output string begins with the number of atoms (rows) in the DataFrame,
    followed by one line per atom containing the symbol ``Ni`` and the selected
    column values separated by spaces.

    :param input_df: 
        Input DataFrame containing atomic or particle data. Each row corresponds to one atom.
    :type input_df: pandas.DataFrame

    :param core: 
        List of column names that define the base ordering of data to be written. 
        Defaults to ``['z', 'y', 'x', 'oz', 'oy', 'ox']``.
    :type core: list[str], optional

    :param extend_col: 
        Additional column or columns to append to the output.
        - If a string is provided, it is appended to ``core``.
        - If a list or NumPy array is provided, its elements are appended to ``core``.
    :type extend_col: str | list[str] | numpy.ndarray, optional

    :returns: 
        A formatted string representing the contents of an ``.xyz`` file.
    :rtype: str

    :notes:
        * The function prepends the total row count on the first line.
        * Each subsequent line starts with the hardcoded atom label ``Ni``.
        * The list ``core`` is modified in place when ``extend_col`` is provided. 
          Pass a copy of the list if you need to preserve the original.

    :example:
        >>> df = pd.DataFrame({
        ...     'x': [0.0, 1.0],
        ...     'y': [0.1, 1.1],
        ...     'z': [0.2, 1.2],
        ...     'extra': [42, 43]
        ... })
        >>> print(convert_xyz(df, core=['z', 'y', 'x'], extend_col='extra'))
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