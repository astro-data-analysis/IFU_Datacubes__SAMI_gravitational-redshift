"""Read SAMI annular FITS products without altering their spectra."""

from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.io import fits

from ..data_models import AnnularCube


def load_sami_annular_cube(path: str | Path) -> AnnularCube:
    """Load one SAMI annular cube from a FITS or compressed FITS file.

    Read flux from PRIMARY, variance from VARIANCE, and region labels from
    BIN_MASK. Reconstruct the linear wavelength axis using the same formula
    as the reference notebook::

        wavelength = CRVAL3 + (pixel_number - CRPIX3) * CDELT3

    FITS pixel numbers start at one. The returned arrays are independent
    copies, so they remain available after the FITS file is closed.

    Parameters
    ----------
    path : str or pathlib.Path
        Local path to a SAMI annular blue or red cube. Both .fits and
        .fits.gz files are supported by Astropy.

    Returns
    -------
    AnnularCube
        Original flux and variance arrays with shape (n_wave, ny, nx),
        wavelength array with shape (n_wave,), and BIN_MASK with shape
        (ny, nx). Wavelength values retain the unit in CUNIT3. Z_SPEC is
        stored as z_initial when present, otherwise as None. NAME is used
        as the galaxy identifier, falling back to OBJECT.

        CTYPE3=AWAV identifies air wavelengths; WAVE identifies vacuum
        wavelengths. SPECSYS is preserved when present; otherwise the
        spectral reference frame remains unknown. The arm is inferred
        only from standard *_annular_blue.fits[.gz] or
        *_annular_red.fits[.gz] filenames; otherwise it remains None.

    Raises
    ------
    OSError
        If the file cannot be opened.
    KeyError
        If a required FITS extension or spectral keyword is missing.
    ValueError
        If array dimensions are inconsistent, the wavelength axis is not
        a supported linear wavelength axis, its unit is not a length,
        or a provided Z_SPEC is invalid.

    Notes
    -----
    This loader targets the linear, separable spectral axis of the SAMI. 
    Other WCS formats require a dedicated reader. It performs basic 
    structural checks only: it does not mask invalid flux or variance values, 
    normalize, resample, fit lines, or extract annuli. BIN_MASK labels are 
    preserved, including the agreed central region 1 and outer region 4.

    Flux units and the original headers are retained as supplied. Detailed
    uncertainty-unit validation belongs to the validation stage.
    """
    path = Path(path).expanduser()

    with fits.open(path, memmap=False) as hdul:
        for extension in (0, "VARIANCE", "BIN_MASK"):
            if hdul[extension].data is None:
                raise ValueError(f"FITS extension {extension!r} contains no data.")

        flux = np.array(hdul[0].data, copy=True)
        variance = np.array(hdul["VARIANCE"].data, copy=True)
        bin_mask = np.array(hdul["BIN_MASK"].data, copy=True)
        header = hdul[0].header.copy()
        variance_header = hdul["VARIANCE"].header.copy()
        bin_header = hdul["BIN_MASK"].header.copy()

    if flux.ndim != 3 or variance.shape != flux.shape:
        raise ValueError("Flux and variance must have matching (n_wave, ny, nx) shapes.")
    if bin_mask.shape != flux.shape[1:]:
        raise ValueError("BIN_MASK must match the spatial shape of the flux cube.")

    ctype = str(header["CTYPE3"]).strip().upper()
    if ctype not in ("AWAV", "WAVE"):
        raise ValueError(f"Unsupported spectral axis: CTYPE3={ctype!r}.")
    # Do not silently apply the notebook formula to a different WCS layout.
    if any(key.startswith("CD3_") or key in ("CD1_3", "CD2_3") for key in header):
        raise ValueError("Spectral CD matrix terms require a dedicated WCS reader.")
    for key, expected in (("PC3_1", 0), ("PC3_2", 0), ("PC1_3", 0),
                          ("PC2_3", 0), ("PC3_3", 1)):
        if float(header.get(key, expected)) != expected:
            raise ValueError("Nontrivial spectral PC matrix terms are not supported.")

    wavelength_unit = str(header["CUNIT3"]).strip()
    if not u.Unit(wavelength_unit).is_equivalent(u.m):
        raise ValueError("CUNIT3 must describe a wavelength unit.")
    crval = float(header["CRVAL3"])
    crpix = float(header["CRPIX3"])
    cdelt = float(header["CDELT3"])
    if not np.all(np.isfinite([crval, crpix, cdelt])) or cdelt == 0:
        raise ValueError("Spectral coordinates must be finite with a nonzero CDELT3.")
    pixel_number = np.arange(flux.shape[0], dtype=float) + 1
    wavelength = crval + (pixel_number - crpix) * cdelt
    if not np.all(np.isfinite(wavelength) & (wavelength > 0)):
        raise ValueError("Reconstructed wavelengths must be finite and positive.")

    z_initial = header.get("Z_SPEC")
    if z_initial is not None:
        z_initial = float(z_initial)
        if not np.isfinite(z_initial) or z_initial <= -1:
            raise ValueError("Z_SPEC must be finite and greater than -1.")

    galaxy_id = header.get("NAME", header.get("OBJECT"))
    filename = path.name.lower().removesuffix(".gz").removesuffix(".fits")
    arm = None
    if filename.endswith("_annular_blue"):
        arm = "blue"
    elif filename.endswith("_annular_red"):
        arm = "red"

    return AnnularCube(
        flux=flux,
        variance=variance,
        wavelength=wavelength,
        bin_mask=bin_mask,
        wavelength_unit=wavelength_unit,
        flux_unit=header.get("BUNIT"),
        z_initial=z_initial,
        galaxy_id=str(galaxy_id) if galaxy_id is not None else None,
        arm=arm,
        wavelength_convention="air" if ctype == "AWAV" else "vacuum",
        spectral_frame=str(header.get("SPECSYS", "unknown")),
        metadata={
            "source_path": str(path.resolve()),
            "primary_header": header,
            "variance_header": variance_header,
            "bin_mask_header": bin_header,
        },
    )
