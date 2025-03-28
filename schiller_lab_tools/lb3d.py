#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import numpy as np
import pandas as pd


# In[2]:


def read_hdf5(filename):
    """
    Reads data from an HDF5 file and returns it as a NumPy array.

    The function attempts to read the first dataset in the specified HDF5 file
    and converts it into a NumPy array in Fortran order. If the file cannot 
    be read or an error occurs during the process, the function returns 0.

    :param filename: 
        The absolute or relative path to the HDF5 file to be read.
    :type filename: str

    :return: 
        A NumPy array containing the data from the first dataset in the file,
        arranged in Fortran order, if the file is successfully read. Returns 0 if 
        the file cannot be read or an error occurs.
    :rtype: numpy.ndarray or int

    :note: 
        - This function assumes the first dataset in the HDF5 file is to be read.
        - The dataset is loaded in its native layout, and no reshaping or 
          additional processing is performed.

    :example:
        >>> import numpy as np
        >>> data = read_hdf5('example.h5')
        >>> if isinstance(data, np.ndarray):
        >>>     print("Data shape:", data.shape)
        >>> else:
        >>>     print("Failed to read the file.")
    """
    try:
        with h5py.File(filename, 'r') as file:
            OutArray = file.get(list(file.keys())[0])
            data = np.array(OutArray)
    except:
        data = 0
    return data


# In[3]:


def read_asc(path, headers=None):
    """
    Reads data from an ASC file and returns it as a pandas DataFrame along with the timestep.

    This function processes an ASC file containing simulation data. If no `headers` 
    are provided, default headers are applied based on the length of the data. 
    If the file length is 0, a default DataFrame with zero-filled columns is generated. 

    :param path: 
        The file path of the ASC file to be read.
    :type path: str

    :param headers: 
        Custom column headers for the output DataFrame. If not provided, default 
        headers are assigned based on the length of the data.
    :type headers: list of str, optional

    :return: 
        A tuple containing:
        - md_df (pandas.DataFrame): The DataFrame containing the data from the ASC file.
        - t (int): The timestep extracted from the file name.
    :rtype: tuple

    :note: 
        - The function infers the timestep `t` from the file name by extracting the value after 
          the last occurrence of 't' in the file name.
        - Default headers are:
          - `['x', 'y', 'z', 'v_x', 'v_y', 'v_z', 'o_x', 'o_y', 'o_z', 'w_x', 'w_y', 'w_z', 'p_id']` 
            for files with 13 columns.
          - `['x', 'y', 'z', 'v_x', 'v_y', 'v_z', 'o_x', 'o_y', 'o_z', 'w_x', 'w_y', 'w_z', 
            'Fb_x', 'Fb_y', 'Fb_z', 't_x', 't_y', 't_z', 'p_id']` for files with more than 13 columns.
        - If the ASC file is empty, a default DataFrame with zero-filled columns and default headers 
          is returned.

    :example:
        >>> df, t = read_asc('data/asc_file_t100.asc')
        >>> print(df.head())
        >>> print("Timestep:", t)

        >>> custom_headers = ['col1', 'col2', 'col3']
        >>> df, t = read_asc('data/asc_file_t100.asc', headers=custom_headers)
        >>> print(df.head())
        >>> print("Timestep:", t)
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


# In[4]:


def rewrite_asc_file(path):
    """
    Rewrites an ASC file to ensure it is in a NumPy-parsable format.

    This function processes an ASC file to correct formatting issues where values 
    representing zero are not written in a parsable scientific notation. The corrected 
    file is saved with the same name, overwriting the original. This ensures compatibility 
    with NumPy parsing, particularly for files generated by lb3d upon checkpoint restart 
    or run start.

    :param path: 
        The absolute or relative path of the ASC file to be rewritten.
    :type path: str

    :return: 
        Returns 1 to indicate the file has been successfully rewritten.
    :rtype: int

    :note: 
        - The function reads the file line by line and checks each numeric value in the file.
        - If a value is not in a parsable scientific notation (e.g., missing 'E'), it is corrected.
        - The corrected file is written back to the same path, overwriting the original.

    :example:
        >>> rewrite_asc_file('output/asc_file.asc')
        1

        After execution, the file `asc_file.asc` will be in the correct format for NumPy parsing.
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

