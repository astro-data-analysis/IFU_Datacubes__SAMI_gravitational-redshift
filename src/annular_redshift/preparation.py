"""Automatically prepare individual emission lines on the original spectral grid.

This adaptive procedure extends the fixed Halpha windows in the notebook.
Its numerical rules are implementation choices, not parameters specified by
Padilla et al. Preliminary fits select windows; they are not final measurements.
"""

from copy import deepcopy

import numpy as np
from astropy import units as u
from scipy.optimize import least_squares

from .curation import build_valid_mask
from .data_models import Spectrum

C_KM_S = 299792.458


def predict_line_position(spectrum: Spectrum, catalog: dict, line_id: str) -> float:
    """Return the expected observed wavelength in the spectrum's own unit.

    Use the catalogue rest wavelength and spectrum.z_initial. Air/vacuum
    conventions must already match; only wavelength units are converted.
    A missing redshift, unknown/mismatched convention, disabled line, or
    nonpositive/nonfinite reference raises ValueError. No data are shifted.
    The initial redshift must refer to the frame of the supplied spectrum.
    """
    line = catalog['lines'][line_id]
    if not line['enabled'] or line['rest_wavelength'] is None:
        raise ValueError(f'{line_id}: line is disabled or its reference is undefined.')
    if (spectrum.wavelength_convention not in ('air', 'vacuum')
            or spectrum.wavelength_convention != catalog['wavelength_convention']):
        raise ValueError('Spectrum and catalogue must have matching known wavelength conventions.')
    z = spectrum.z_initial
    if z is None or not np.isfinite(z) or z <= -1:
        raise ValueError('A finite initial redshift greater than -1 is required.')
    rest = (line['rest_wavelength'] * u.Unit(catalog['wavelength_unit'])).to_value(
        u.Unit(spectrum.wavelength_unit))
    expected = rest * (1 + z)
    if not np.isfinite(expected) or expected <= 0:
        raise ValueError('Expected wavelength must be finite and positive.')
    return float(expected)


def build_line_windows(
    spectrum: Spectrum, expected: float, center: float, sigma: float,
    neighbor_positions: np.ndarray, neighbor_guard: float, *,
    fit_sigma: float = 4.0, continuum_inner_sigma: float = 5.0,
    continuum_outer_sigma: float = 12.0,
) -> dict:
    """Build adaptive fit and continuum masks in the spectrum's wavelength unit.

    True means SELECTED in these masks, unlike Spectrum.mask. Window sizes
    are multiples of the preliminary observed Gaussian sigma. Neighbouring
    catalogue transitions are excluded from continuum bands even when they
    are disabled as measurement targets. Fit limits cannot cross the midpoint
    between the expected target position and the nearest neighbouring line.
    No channel is deleted, interpolated or aligned. Negative flux is retained.

    Return bounds, masks and a flag indicating neighbor-clipped fit bounds.
    Continuum is estimated from both sides. Insufficient usable channels are
    handled by prepare_line, not by silently widening across neighbouring lines.
    These generic rules are configurable implementation choices.
    """
    values = [expected, center, sigma, neighbor_guard, fit_sigma,
              continuum_inner_sigma, continuum_outer_sigma]
    if not np.all(np.isfinite(values)) or min(sigma, neighbor_guard, fit_sigma) <= 0:
        raise ValueError('Window parameters must be finite and widths positive.')
    if not fit_sigma < continuum_inner_sigma < continuum_outer_sigma:
        raise ValueError('Require fit_sigma < continuum_inner_sigma < continuum_outer_sigma.')
    wave = spectrum.wavelength
    valid = build_valid_mask(spectrum)
    neighbors = np.asarray(neighbor_positions, dtype=float)
    if neighbors.ndim != 1 or not np.all(np.isfinite(neighbors)):
        raise ValueError('Neighbor positions must be a finite one-dimensional array.')
    left = neighbors[neighbors < expected]
    right = neighbors[neighbors > expected]
    lower = center - fit_sigma * sigma
    upper = center + fit_sigma * sigma
    fit_lower = max(lower, (left.max() + expected) / 2) if left.size else lower
    fit_upper = min(upper, (right.min() + expected) / 2) if right.size else upper
    blue = (center - continuum_outer_sigma * sigma, center - continuum_inner_sigma * sigma)
    red = (center + continuum_inner_sigma * sigma, center + continuum_outer_sigma * sigma)
    blue_mask = valid & (wave >= blue[0]) & (wave <= blue[1])
    red_mask = valid & (wave >= red[0]) & (wave <= red[1])
    for neighbor in neighbors:
        allowed = np.abs(wave - neighbor) > neighbor_guard
        blue_mask &= allowed
        red_mask &= allowed
    return {
        'fit_mask': valid & (wave >= fit_lower) & (wave <= fit_upper),
        'continuum_mask': blue_mask | red_mask,
        'blue_continuum_mask': blue_mask,
        'red_continuum_mask': red_mask,
        'fit_bounds': (float(fit_lower), float(fit_upper)),
        'continuum_bounds': (blue, red),
        'fit_clipped_by_neighbor': bool(fit_lower != lower or fit_upper != upper),
    }


def fit_continuum(spectrum: Spectrum, mask: np.ndarray, reference: float) -> dict:
    """Fit a weighted straight continuum using selected valid channels.

    mask=True selects continuum channels. Fit intercept + slope*(wave-reference)
    with inverse-variance weights. Return the continuum on the full original
    grid, coefficients and their conditional covariance (known input variance).
    This covariance is not a complete continuum-systematic uncertainty model.
    At least three selected channels with a nonsingular design are required.
    """
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.shape != spectrum.flux.shape:
        raise ValueError('Continuum mask must be boolean and match the spectrum.')
    selected = mask & build_valid_mask(spectrum)
    if selected.sum() < 3 or not np.isfinite(reference):
        raise ValueError('Continuum requires at least three valid channels and a finite reference.')
    x = spectrum.wavelength[selected] - reference
    # Scale coordinates to keep small/large wavelength units numerically stable.
    x_scale = float(np.max(np.abs(x)))
    if x_scale == 0:
        raise ValueError('Continuum coordinates have no wavelength span.')
    error = np.sqrt(spectrum.variance[selected])
    design = np.column_stack([np.ones(x.size), x / x_scale])
    weighted = design / error[:, None]
    coefficients, _, rank, _ = np.linalg.lstsq(weighted, spectrum.flux[selected] / error, rcond=None)
    if rank != 2:
        raise ValueError('Continuum fit is singular.')
    conversion = np.diag([1.0, 1.0 / x_scale])
    covariance = conversion @ np.linalg.inv(weighted.T @ weighted) @ conversion
    intercept, slope = conversion @ coefficients
    return {'continuum': intercept + slope * (spectrum.wavelength - reference),
            'intercept': float(intercept), 'slope': float(slope),
            'reference': float(reference), 'covariance': covariance}


def prepare_line(
    spectrum: Spectrum, catalog: dict, line_id: str, *,
    instrumental_sigma_kms: float,
    search_half_width_kms: float = 600.0,
    initial_width_factor: float = 3.0,
    max_iterations: int = 8,
    convergence_fraction: float = 0.05,
    min_continuum_channels_per_side: int = 5,
    min_fit_channels: int = 7,
) -> dict:
    """Automatically choose windows, estimate continuum and initialize a line fit.

    Parameters
    ----------
    spectrum : Spectrum
        Extracted spectrum, optionally curated. Its grid and values are preserved.
    catalog : dict
        Validated output of load_line_catalog.
    line_id : str
        Enabled individual-line identifier, such as 'Halpha'.
    instrumental_sigma_kms : float
        Positive instrumental Gaussian SIGMA in km/s at the target line,
        not FWHM or resolving power. Supply it from the instrument calibration;
        it is not inferred from a filename. Use consistent inputs for both annuli.
    search_half_width_kms : float
        General maximum center-search offset from the catalogue prediction.
    initial_width_factor : float
        Initial observed width in multiples of instrumental/sampling width.
    max_iterations : int
        Maximum adaptive continuum/window iterations.
    convergence_fraction : float
        Convergence tolerance for center and width changes, relative to width.
    min_continuum_channels_per_side, min_fit_channels : int
        Minimum usable channel counts; these are numerical requirements, not
        signal-to-noise detection thresholds.

    Returns
    -------
    dict
        Masks (True=selected), continuum and continuum-subtracted flux on the
        original grid, preliminary amplitude/center/sigma, parameter bounds,
        continuum fit, provenance and options. status='prepared' does NOT
        certify detection, unimodality or a final redshift measurement.

    Raises
    ------
    ValueError
        For unsupported features/conventions, invalid inputs, insufficient
        coverage, failed convergence, boundary-limited profiles or insufficient
        channels. A future pipeline should record the failure and continue.

    Notes
    -----
    A preliminary weighted Gaussian fit determines the window scale; final
    measurements and their uncertainties belong to fitting.py. Flux/error are
    scaled together internally for numerical stability, including negative
    channels. Fit bounds use the same expected reference for all annuli. No
    spectrum is shifted to its preliminary center. Continuum bands start at 5-12
    sigmas and expand automatically up to 48 sigmas if channel counts require
    it. The fit window is +/-4 sigmas. Neighbour guards
    use the larger of three instrumental sigmas and four estimated target sigmas.
    These adaptive rules extend, rather than exactly reproduce, the notebook's
    fixed windows. Broad neighbours, stellar absorption and unlisted lines can
    still contaminate the continuum; final diagnostics must assess that risk.
    """
    for name, value in [('instrumental_sigma_kms', instrumental_sigma_kms),
                        ('search_half_width_kms', search_half_width_kms),
                        ('initial_width_factor', initial_width_factor),
                        ('convergence_fraction', convergence_fraction)]:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive.')
    for value, minimum in [(max_iterations, 1), (min_fit_channels, 4),
                           (min_continuum_channels_per_side, 2)]:
        if type(value) is not int or value < minimum:
            raise ValueError('Iteration/channel limits must be valid positive integers.')
    if catalog['lines'][line_id]['feature_type'] != 'individual_line':
        raise ValueError('This preparation method supports individual lines only.')
    expected = predict_line_position(spectrum, catalog, line_id)
    wave = spectrum.wavelength
    valid = build_valid_mask(spectrum)
    if not np.all(np.isfinite(wave)) or not (np.all(np.diff(wave) > 0) or np.all(np.diff(wave) < 0)):
        raise ValueError('A finite, strictly monotonic wavelength grid is required.')
    if not wave.min() < expected < wave.max():
        raise ValueError('Expected line is outside usable spectral coverage.')
    step = float(np.median(np.abs(np.diff(wave))))
    instrument_sigma = expected * instrumental_sigma_kms / C_KM_S
    min_sigma = max(instrument_sigma, step / 2)
    search_width = expected * search_half_width_kms / C_KM_S
    max_sigma = search_width / 2
    if max_sigma <= min_sigma:
        raise ValueError('Search range is too narrow for the instrumental/sampling width.')
    conversion = u.Unit(catalog['wavelength_unit']).to(u.Unit(spectrum.wavelength_unit))
    neighbors = []
    for key, line in catalog['lines'].items():
        if key == line_id:
            continue
        references = line.get('components', [line.get('rest_wavelength')])
        neighbors.extend(value * conversion * (1 + spectrum.z_initial)
                         for value in references if value is not None)
    neighbors = np.asarray(neighbors)
    left, right = neighbors[neighbors < expected], neighbors[neighbors > expected]
    lower_center = max(expected-search_width, (left.max()+expected)/2) if left.size else expected-search_width
    upper_center = min(expected+search_width, (right.min()+expected)/2) if right.size else expected+search_width
    center = expected
    margin = .01 * (max_sigma - min_sigma)
    sigma = float(np.clip(initial_width_factor * min_sigma, min_sigma + margin, max_sigma - margin))
    converged = False
    for iteration in range(max_iterations):
        for outer_factor in (12., 18., 27., 40.5, 48.):
            windows = build_line_windows(
                spectrum, expected, center, sigma, neighbors,
                max(3 * instrument_sigma, 4 * sigma),
                continuum_outer_sigma=outer_factor)
            if all(windows[key].sum() >= min_continuum_channels_per_side
                   for key in ('blue_continuum_mask', 'red_continuum_mask')):
                break
        else:
            raise ValueError('Insufficient clean continuum channels on both sides.')
        if windows['fit_mask'].sum() < min_fit_channels:
            raise ValueError('Insufficient valid line channels.')
        continuum = fit_continuum(spectrum, windows['continuum_mask'], expected)
        selected = windows['fit_mask'] & valid
        residual_flux = spectrum.flux[selected] - continuum['continuum'][selected]
        error = np.sqrt(spectrum.variance[selected])
        scale = max(float(np.max(np.abs(residual_flux))), float(np.median(error)))
        x = (wave[selected] - expected) / search_width
        flux, err = residual_flux / scale, error / scale
        def residual(parameters):
            amplitude, offset, width = parameters
            return (amplitude * np.exp(-.5 * ((x-offset)/width)**2) - flux) / err
        lower = [0, (lower_center-expected)/search_width, min_sigma/search_width]
        upper = [np.inf, (upper_center-expected)/search_width, max_sigma/search_width]
        initial = [max(float(np.max(flux)), 1e-6), (center-expected)/search_width, sigma/search_width]
        result = least_squares(residual, initial, bounds=(lower, upper), max_nfev=2000)
        if not result.success or not np.all(np.isfinite(result.x)):
            raise ValueError('Preliminary Gaussian fit failed.')
        if result.active_mask[0] or result.active_mask[1] or result.active_mask[2] == 1:
            raise ValueError('Preliminary profile is limited by amplitude, center or maximum-width bounds.')
        amplitude = float(result.x[0] * scale)
        new_center = float(expected + result.x[1] * search_width)
        new_sigma = float(result.x[2] * search_width)
        if max(abs(new_center-center), abs(new_sigma-sigma)) <= convergence_fraction * sigma:
            converged = True
            break
        center, sigma = new_center, new_sigma
    if not converged:
        raise ValueError('Automatic window selection did not converge.')
    return {
        'status': 'prepared', 'line_id': line_id, 'galaxy_id': spectrum.galaxy_id,
        'region_id': spectrum.region_id, 'expected_wavelength': expected,
        'wavelength_unit': spectrum.wavelength_unit,
        'wavelength_convention': spectrum.wavelength_convention,
        'continuum': continuum['continuum'],
        'flux_without_continuum': spectrum.flux - continuum['continuum'],
        'continuum_fit': continuum,
        'initial_parameters': {'amplitude': amplitude, 'center': new_center, 'sigma': new_sigma},
        'parameter_bounds': {'amplitude': (0., np.inf), 'center': (lower_center, upper_center),
                             'sigma': (min_sigma, max_sigma)},
        'iterations': iteration+1,
        'window_reference': {'center': center, 'sigma': sigma},
        'options': {'instrumental_sigma_kms': instrumental_sigma_kms,
                    'search_half_width_kms': search_half_width_kms,
                    'initial_width_factor': initial_width_factor,
                    'max_iterations': max_iterations, 'convergence_fraction': convergence_fraction,
                    'min_continuum_channels_per_side': min_continuum_channels_per_side,
                    'min_fit_channels': min_fit_channels,
                    'fit_sigma': 4., 'continuum_inner_sigma': 5., 'continuum_outer_sigma': outer_factor,
                    'continuum_outer_factors': (12., 18., 27., 40.5, 48.)},
        'catalog_id': catalog['catalog_id'], 'catalog_line': deepcopy(catalog['lines'][line_id]),
        'spectrum_metadata': deepcopy(spectrum.metadata),
        **windows,
    }
