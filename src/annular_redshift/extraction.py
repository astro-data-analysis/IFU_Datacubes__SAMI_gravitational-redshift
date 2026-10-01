"""Extract representative spectra from already binned annular cubes."""

from copy import deepcopy

import numpy as np

from .data_models import AnnularCube, Spectrum


def check_bin_spectra_consistency(
    cube: AnnularCube,
    region_id: int,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> dict:
    """Check whether a bin repeats the same flux and variance spectra.

    Parameters
    ----------
    cube : AnnularCube
        Already binned cube with arrays ordered as (wavelength, y, x).
    region_id : int
        Original BIN_MASK label: 1 for the central region or 4 for the
        outer region used in the reference notebook.
    rtol, atol : float, optional
        Nonnegative, finite relative and absolute comparison tolerances.
        Defaults match the notebook's numpy.allclose defaults. Comparisons
        use the original numerical arrays; atol therefore applies in their
        respective stored flux and variance units. No scaling is performed.

    Returns
    -------
    dict
        Region ID, channel and spaxel counts, representative coordinates,
        separate consistency flags and mismatch counts for flux and variance,
        a combined flag, and the comparison tolerances used.
        Mismatch counts use the same tolerances as the consistency flags.

    Raises
    ------
    ValueError
        If shapes or tolerances are invalid, region_id is not an integer,
        or the requested bin has no spaxels.

    Notes
    -----
    Each spectrum is compared with the first spaxel returned by argwhere,
    following the notebook's (y, x) ordering. NaNs in matching positions
    compare as equal. This verifies repetition, not scientific usability:
    identical spectra can still contain invalid channels, even all NaNs.
    Channel selection belongs to the subsequent curation stage.
    """
    if isinstance(region_id, (bool, np.bool_)) or not isinstance(
        region_id, (int, np.integer)
    ):
        raise ValueError("region_id must be an integer BIN_MASK label.")
    if not np.all(np.isfinite([rtol, atol])) or rtol < 0 or atol < 0:
        raise ValueError("rtol and atol must be finite and nonnegative.")
    if cube.flux.ndim != 3 or cube.variance.shape != cube.flux.shape:
        raise ValueError("Flux and variance must have matching 3D shapes.")
    if cube.bin_mask.shape != cube.flux.shape[1:]:
        raise ValueError("BIN_MASK must match the cube's spatial dimensions.")
    if cube.wavelength.shape != (cube.flux.shape[0],) or not cube.flux.shape[0]:
        raise ValueError("Wavelength must be a nonempty array matching the spectral axis.")

    coordinates = np.argwhere(cube.bin_mask == region_id)
    if len(coordinates) == 0:
        raise ValueError(f"No spaxels found for BIN_MASK={region_id}.")
    y_rep, x_rep = map(int, coordinates[0])
    selected = cube.bin_mask == region_id
    flux_spectra = cube.flux[:, selected]
    variance_spectra = cube.variance[:, selected]

    flux_matches = np.all(
        np.isclose(flux_spectra, flux_spectra[:, :1],
                   rtol=rtol, atol=atol, equal_nan=True), axis=0
    )
    variance_matches = np.all(
        np.isclose(variance_spectra, variance_spectra[:, :1],
                   rtol=rtol, atol=atol, equal_nan=True), axis=0
    )
    return {
        "region_id": int(region_id),
        "n_channels": int(cube.flux.shape[0]),
        "n_spaxels": int(len(coordinates)),
        "y_rep": y_rep,
        "x_rep": x_rep,
        "flux_consistent": bool(np.all(flux_matches)),
        "variance_consistent": bool(np.all(variance_matches)),
        "consistent": bool(np.all(flux_matches) and np.all(variance_matches)),
        "n_flux_mismatches": int(np.count_nonzero(~flux_matches)),
        "n_variance_mismatches": int(np.count_nonzero(~variance_matches)),
        "rtol": float(rtol),
        "atol": float(atol),
    }


def extract_annular_spectrum(
    cube: AnnularCube,
    region_id: int,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> Spectrum:
    """Extract the first representative spaxel after checking bin repetition.

    Parameters
    ----------
    cube : AnnularCube
        Cube returned by the SAMI loader.
    region_id : int
        Original BIN_MASK label. Use 1 for the central region and 4 for
        the outer region, as agreed from the reference notebook.
    rtol, atol : float, optional
        Tolerances forwarded to check_bin_spectra_consistency.

    Returns
    -------
    Spectrum
        Independent copies of wavelength, flux, variance and metadata.
        Units, initial redshift, galaxy ID and spectral conventions are
        preserved. metadata['extraction'] records the representative spaxel,
        bin size and consistency check. mask is None until curation.
        Standard errors are available through the Spectrum.error property.

    Raises
    ------
    ValueError
        If the bin is absent, input structure is invalid, or flux or variance
        spectra differ beyond the specified tolerances within the bin.

    Notes
    -----
    These spectra have already been combined in the input product. They
    are neither summed nor averaged again, and their variance is not divided
    by the number of repeated spaxels. No normalization, redshift correction,
    line selection or filtering of invalid channels is performed here.
    """
    report = check_bin_spectra_consistency(cube, region_id, rtol=rtol, atol=atol)
    if not report["consistent"]:
        raise ValueError(
            f"BIN_MASK={region_id} does not contain repeated spectra: "
            f"{report['n_flux_mismatches']} flux and "
            f"{report['n_variance_mismatches']} variance mismatches."
        )

    y_rep, x_rep = report["y_rep"], report["x_rep"]
    metadata = deepcopy(cube.metadata)
    metadata["extraction"] = {"method": "first_repeated_spaxel", **report}
    return Spectrum(
        wavelength=cube.wavelength.copy(),
        flux=cube.flux[:, y_rep, x_rep].copy(),
        variance=cube.variance[:, y_rep, x_rep].copy(),
        region_id=int(region_id),
        wavelength_unit=cube.wavelength_unit,
        flux_unit=cube.flux_unit,
        z_initial=cube.z_initial,
        galaxy_id=cube.galaxy_id,
        arm=cube.arm,
        wavelength_convention=cube.wavelength_convention,
        spectral_frame=cube.spectral_frame,
        metadata=metadata,
    )
