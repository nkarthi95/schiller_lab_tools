import numpy as np
import scipy.fftpack as fft

def structure_factor(phi):
    """
    Compute the structure factor of a given field using its Fourier transform.

    This function calculates the structure factor of a 2D or 3D field by performing 
    a Fourier transform and computing the intensity at each wavevector `k`. The 
    structure factor provides insight into the spatial frequency content of the field.

    :param phi:
        A 2D or 3D array representing the field whose structure factor is to be computed. 
        The input array should be of real or complex values.
    :type phi: numpy.ndarray

    :return:
        A tuple containing:
        - **k** (*numpy.ndarray*): The wavevectors corresponding to the Fourier transform 
          frequencies. This is a meshgrid of wavevectors along each dimension of the input field.
        - **S** (*numpy.ndarray*): The structure factor, which is the intensity of the 
          Fourier transform at each wavevector `k`. This is calculated as the square of the 
          magnitude of the Fourier transform, normalized by the number of elements in the input array.
    :rtype: tuple

    :note:
        - The Fourier transform is computed using `fft.fftn`, and the field is mean-centered 
          before the transform to remove any zero-frequency component.
        - The structure factor is computed as `S(k) = (1/N) * |F(phi)(k)|^2`, where `N` is 
          the number of elements in the field and `F(phi)(k)` is the Fourier transform of the field.
        - The result is the distribution of intensities across different spatial frequencies.

    :example:
        >>> import numpy as np
        >>> phi = np.random.random((10, 10))
        >>> k, S = structure_factor(phi)
        >>> print(k.shape, S.shape)
    """
    k = np.stack(np.meshgrid(*tuple([2*np.pi*fft.fftfreq(L) for L in phi.shape]), indexing='ij'), axis=-1)
    N = np.prod(phi.shape)
    phi_k = fft.fftn(phi - np.mean(phi))
    S = 1 / N * np.abs(phi_k) ** 2
    return k, S

def spherically_averaged_structure_factor(phi, binned=True):
    """
    Compute the spherically averaged structure factor of a field.

    This function calculates the structure factor of a given field using its Fourier transform 
    and averages the intensities over spherical shells in wavevector space. It provides a way 
    to analyze the frequency distribution of the field's spatial structure. The function can 
    return either a binned or unbinned version of the structure factor.

    :param phi:
        A 2D or 3D array representing the field whose structure factor is to be computed. 
        The input array should be of real or complex values.
    :type phi: numpy.ndarray

    :param binned:
        A boolean flag indicating whether to return the binned version of the structure factor. 
        If `True`, the result will be averaged over spherical bins. If `False`, the unbinned 
        version is returned. Default is `True`.
    :type binned: bool, optional

    :return:
        A tuple containing:
        - **k** (*numpy.ndarray*): The wavevectors corresponding to the Fourier transform 
          frequencies, binned or unbinned, depending on the `binned` flag.
        - **S** (*numpy.ndarray*): The structure factor values at each wavevector `k`, either 
          binned or unbinned.
    :rtype: tuple

    :note:
        - The function first computes the structure factor using `structure_factor`, then computes 
          the spherical radial distances from the wavevector grid.
        - If `binned` is `True`, the structure factor is averaged over spherical shells of 
          wavevectors, with each bin corresponding to a range of wavevector magnitudes.
        - If `binned` is `False`, the function returns the structure factor values for each 
          individual wavevector.
        - The radial distance from the origin in wavevector space is calculated using the 
          Euclidean norm of the wavevector `k`.

    :example:
        >>> import numpy as np
        >>> phi = np.random.random((10, 10))
        >>> k, S = spherically_averaged_structure_factor(phi, binned=True)
        >>> print(k.shape, S.shape)

        >>> k, S = spherically_averaged_structure_factor(phi, binned=False)
        >>> print(k.shape, S.shape)
    """
    L = np.min(np.shape(phi))
    k, S = structure_factor(phi)
    k1 = np.linalg.norm(k, axis=-1).flatten()
    S1 = S.flatten()
    kmin = 2*np.pi/L  # sampling frequency
    where = np.s_[:]  # Select all k values (no upper bound)
    
    if not binned:
        ku, inv, counts = np.unique(k1[where], return_inverse=True, return_counts=True)
        sums = np.zeros(len(ku), dtype=S1.dtype)
        np.add.at(sums, inv, S1[where])
        return ku, sums/counts
    else:
        bins = np.arange(L//2+1) * kmin  # Define bin edges
        shells = np.histogram(k1[where], bins, weights=S1[where])[0]
        counts = np.histogram(k1[where], bins)[0]
        return (bins[:-1] + bins[1:]) / 2, shells / counts
    
def spherical_first_moment(phi, binned=True):
    """
    Compute the first moment of the spherically averaged structure factor.

    This function calculates the first moment of the structure factor, which is used to 
    determine the characteristic length scale of the input order parameter `phi`. The 
    structure factor is first computed and optionally binned in spherical shells. The first 
    moment is then computed as the weighted average of the wavevector magnitudes.

    :param phi:
        A 2D or 3D array representing the order parameter of the field. This array typically 
        represents some physical data whose spatial structure is being analyzed.
    :type phi: numpy.ndarray

    :param binned:
        A boolean flag indicating whether the structure factor should be averaged over spherical 
        shells. If `True`, the structure factor is binned and averaged. If `False`, the unbinned 
        structure factor is used. Default is `True`.
    :type binned: bool, optional

    :return:
        The first moment of the spherically averaged structure factor, which is used to estimate 
        the characteristic length scale of the field represented by `phi`.
    :rtype: float

    :note:
        - The first moment is calculated as the weighted average of the wavevector magnitudes, with 
          the structure factor values serving as weights.
        - The result of this function can be interpreted as a characteristic length scale of the 
          spatial structure in `phi`.

    :example:
        >>> import numpy as np
        >>> phi = np.random.random((10, 10))
        >>> k1 = spherical_first_moment(phi, binned=True)
        >>> print(k1)

        >>> k1 = spherical_first_moment(phi, binned=False)
        >>> print(k1)
    """
    k, S = spherically_averaged_structure_factor(phi, binned=binned)
    k1 = np.sum(k * S) / np.sum(S)
    return k1

def second_moment(phi):
    """
    Compute the second moment of the structure factor.

    This function calculates the second moment of the structure factor, which is a measure of 
    the spread or variance of the wavevector distribution. The second moment quantifies the 
    spatial distribution of the field's spatial frequencies and returns a 3-element array 
    representing the variance in each spatial direction ([x, y, z]).

    :param phi:
        A 2D or 3D array representing the order parameter of the field. This array represents the 
        spatial configuration of the system whose structure factor is to be analyzed.
    :type phi: numpy.ndarray

    :return:
        A 1D array of floats representing the second moment of the structure factor in the 
        [x, y, z] directions. This represents the variance of the spatial frequencies in each 
        direction.
    :rtype: numpy.ndarray

    :note:
        - The second moment is calculated as a weighted average of the squared wavevectors, with 
          the structure factor values serving as weights.
        - This function uses the output from the `structure_factor` function to calculate the second 
          moment.

    :example:
        >>> import numpy as np
        >>> phi = np.random.random((10, 10))
        >>> k2 = second_moment(phi)
        >>> print(k2)
    """
    k, S = structure_factor(phi)
    k2 = np.sum(k * k * S[..., np.newaxis], axis=(0, 1, 2)) / np.sum(S)
    return k2