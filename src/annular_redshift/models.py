"""Analytical profiles used to fit emission lines."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def gaussian(
    wavelength: ArrayLike, amplitude: float, center: float, sigma: float,
) -> NDArray[np.float64]:
    """Evaluate a single Gaussian without a continuum term.

    Parameters
    ----------
    wavelength : array-like
        Wavelength coordinates. No resampling or frame conversion is done.
    amplitude : float
        Peak height above the continuum, in the same units as spectral flux.
        The emission-line fitter constrains this parameter to be nonnegative.
    center : float
        Line center, in the same wavelength unit as the input coordinates.
    sigma : float
        Strictly positive Gaussian standard deviation in wavelength units,
        not FWHM. This is the observed width; no instrumental deconvolution
        is applied.

    Returns
    -------
    numpy.ndarray
        Gaussian values with the same shape as wavelength.

    Raises
    ------
    ValueError
        If parameters are nonfinite or sigma is not strictly positive.

    Notes
    -----
    This is the same functional form used in the Halpha notebook. Selecting
    this model does not establish that an observed line is unimodal.
    """
    if not np.all(np.isfinite([amplitude, center, sigma])) or sigma <= 0:
        raise ValueError('Gaussian parameters must be finite and sigma positive.')
    wavelength = np.asarray(wavelength, dtype=float)
    return amplitude * np.exp(-0.5 * ((wavelength - center) / sigma) ** 2)
