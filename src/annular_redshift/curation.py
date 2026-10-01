"""Build channel masks for annular spectra without modifying their values."""

from copy import deepcopy
from dataclasses import replace
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from .data_models import Spectrum


def build_valid_mask(spectrum: Spectrum) -> NDArray[np.bool_]:
    """Identify usable channels from numerical validity and an existing mask.

    Parameters
    ----------
    spectrum : Spectrum
        One-dimensional wavelength, flux and variance arrays of equal size.
        An existing spectrum.mask uses True for excluded channels.

    Returns
    -------
    numpy.ndarray of bool
        True for channels with finite wavelength, flux and variance,
        strictly positive variance, and no pre-existing exclusion.
        This is a VALIDITY mask: its convention is the inverse of
        Spectrum.mask. No signal-to-noise or flux-sign cut is applied.

    Raises
    ------
    ValueError
        If arrays are empty, not one-dimensional, or have incompatible
        shapes, or if an existing mask is not boolean with the same shape.

    Notes
    -----
    Positive variance is required for finite inverse-variance weights.
    This matches the notebook's finite, strictly positive error criterion
    for spectral fitting. Finite zero and negative flux values are retained.
    Passing these checks does not establish line detection or fit quality.
    """
    wavelength = np.asarray(spectrum.wavelength)
    flux = np.asarray(spectrum.flux)
    variance = np.asarray(spectrum.variance)
    if wavelength.ndim != 1 or wavelength.size == 0:
        raise ValueError("Wavelength must be a nonempty one-dimensional array.")
    if flux.shape != wavelength.shape or variance.shape != wavelength.shape:
        raise ValueError("Flux, variance and wavelength must have matching 1D shapes.")

    valid = (
        np.isfinite(wavelength)
        & np.isfinite(flux)
        & np.isfinite(variance)
        & (variance > 0)
    )
    if spectrum.mask is not None:
        existing = np.asarray(spectrum.mask)
        if existing.shape != valid.shape or existing.dtype != np.dtype(bool):
            raise ValueError("The existing mask must be boolean and match the spectrum.")
        valid &= ~existing
    return valid


def mask_spectral_intervals(
    wavelength: NDArray[np.floating],
    intervals: Iterable[tuple[float, float]] = (),
) -> NDArray[np.bool_]:
    """Mark channels inside explicitly supplied wavelength intervals.

    Parameters
    ----------
    wavelength : numpy.ndarray
        One-dimensional wavelength array, on its original spectral grid.
    intervals : iterable of (float, float), optional
        Inclusive lower and upper bounds. Values must use the SAME unit
        and reference frame as wavelength. Rest-frame intervals must be
        transformed before this function is called. No intervals are
        excluded by default; no sky-line list is assumed.

    Returns
    -------
    numpy.ndarray of bool
        True inside any supplied interval, meaning excluded. Overlapping
        intervals are combined. Out-of-coverage intervals exclude nothing.
        Nonfinite wavelengths are handled separately by build_valid_mask.

    Raises
    ------
    ValueError
        If wavelength is not one-dimensional or interval bounds are
        nonfinite, reversed, or do not form pairs.
    """
    wavelength = np.asarray(wavelength)
    if wavelength.ndim != 1:
        raise ValueError("Wavelength must be one-dimensional.")
    excluded = np.zeros(wavelength.shape, dtype=bool)
    for interval in intervals:
        bounds = np.asarray(interval, dtype=float)
        if bounds.shape != (2,) or not np.all(np.isfinite(bounds)):
            raise ValueError("Each interval must contain two finite bounds.")
        lower, upper = bounds
        if lower > upper:
            raise ValueError("Interval lower bound must not exceed its upper bound.")
        excluded |= (wavelength >= lower) & (wavelength <= upper)
    return excluded


def curate_spectrum(
    spectrum: Spectrum,
    *,
    excluded_intervals: Iterable[tuple[float, float]] = (),
) -> Spectrum:
    """Return a spectrum copy with numerical and interval exclusions combined.

    Parameters
    ----------
    spectrum : Spectrum
        Extracted annular spectrum. The input object is not modified.
    excluded_intervals : iterable of (float, float), optional
        Inclusive intervals in spectrum.wavelength_unit and the same
        spectral frame as spectrum.wavelength. Empty by default.

    Returns
    -------
    Spectrum
        Independent copies of the original arrays and metadata, with a
        boolean mask (True = excluded). Existing exclusions are preserved.
        metadata['curation_history'] records the criteria, intervals and
        channel counts for each call. No channels are deleted or replaced,
        no interpolation is performed, and no continuum is subtracted.

    Notes
    -----
    A fully excluded spectrum is returned with n_valid=0 in its history.
    The later fitting stage must report insufficient usable channels.
    This stage does not select galaxies, assess unimodality, or reject
    measurements based on the value or significance of a redshift difference.
    """
    intervals = tuple(tuple(pair) for pair in excluded_intervals)
    valid = build_valid_mask(spectrum)
    interval_mask = mask_spectral_intervals(spectrum.wavelength, intervals)
    excluded = ~valid | interval_mask
    metadata = deepcopy(spectrum.metadata)
    history = metadata.setdefault("curation_history", [])
    history.append({
        "criteria": "finite wavelength, flux and variance; variance > 0; preserve existing mask",
        "excluded_intervals": [list(map(float, pair)) for pair in intervals],
        "wavelength_unit": spectrum.wavelength_unit,
        "spectral_frame": spectrum.spectral_frame,
        "n_channels": int(excluded.size),
        "n_invalid_or_previously_excluded": int(np.count_nonzero(~valid)),
        "n_new_interval_exclusions": int(np.count_nonzero(valid & interval_mask)),
        "n_excluded": int(np.count_nonzero(excluded)),
        "n_valid": int(np.count_nonzero(~excluded)),
    })
    return replace(
        spectrum,
        wavelength=spectrum.wavelength.copy(),
        flux=spectrum.flux.copy(),
        variance=spectrum.variance.copy(),
        mask=excluded,
        metadata=metadata,
    )
