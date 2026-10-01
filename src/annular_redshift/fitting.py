"""Fit individual emission lines using automatically prepared spectral windows."""

from copy import deepcopy
import warnings

import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning

from .curation import build_valid_mask
from .data_models import Spectrum
from .models import gaussian

PARAMETER_ORDER = ('amplitude', 'center', 'sigma')


def initialize_parameters(prepared: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and return initial parameters and bounds in physical units.

    Parameters follow (amplitude, center, sigma). Bounds are supplied by
    prepare_line, not guessed again here. Amplitude must have a nonnegative
    lower bound and sigma a strictly positive lower bound. All initial
    values must be finite and inside their bounds. Invalid input raises
    ValueError. No parameters or bounds are silently changed.
    """
    initial = np.array([prepared['initial_parameters'][k] for k in PARAMETER_ORDER], dtype=float)
    bounds = np.array([prepared['parameter_bounds'][k] for k in PARAMETER_ORDER], dtype=float)
    if bounds.shape != (3, 2):
        raise ValueError('Each parameter must have a lower and upper bound.')
    lower, upper = bounds[:, 0], bounds[:, 1]
    if (not np.all(np.isfinite(initial)) or np.any(np.isnan(bounds))
            or np.any(lower >= upper) or np.any(initial < lower)
            or np.any(initial > upper)):
        raise ValueError('Initial parameters or parameter bounds are invalid.')
    if lower[0] < 0 or lower[2] <= 0 or not np.isfinite(lower[2]):
        raise ValueError('Emission amplitude must be nonnegative and sigma strictly positive.')
    return initial, lower, upper


def compute_fit_diagnostics(
    flux: np.ndarray, model: np.ndarray, error: np.ndarray, *, n_parameters: int = 3,
) -> dict:
    """Compute descriptive residual statistics for the selected fit channels.

    Return data-minus-model residuals, residuals divided by their errors,
    chi-square, degrees of freedom and reduced chi-square. The continuum is
    held fixed: degrees of freedom are conditional on that prior estimate.
    Statistics assume independent input channel errors. These values do not
    by themselves establish detection, unimodality or gravitational origin.
    """
    flux, model, error = (np.asarray(a, dtype=float) for a in (flux, model, error))
    if flux.ndim != 1 or flux.shape != model.shape or flux.shape != error.shape:
        raise ValueError('Flux, model and error must have matching 1D shapes.')
    if (type(n_parameters) is not int or n_parameters < 0 or flux.size <= n_parameters
            or not np.all(np.isfinite(flux)) or not np.all(np.isfinite(model))
            or not np.all(np.isfinite(error) & (error > 0))):
        raise ValueError('Diagnostics require finite data, positive errors and positive degrees of freedom.')
    residual = flux - model
    standardized = residual / error
    chi_square = float(standardized @ standardized)
    dof = int(flux.size - n_parameters)
    return {'n_channels': int(flux.size), 'degrees_of_freedom': dof,
            'chi_square': chi_square, 'reduced_chi_square': chi_square / dof,
            'residual': residual, 'standardized_residual': standardized}


def fit_emission_line(spectrum: Spectrum, prepared: dict, *, max_evaluations: int = 10000) -> dict:
    """Fit one Gaussian to a prepared continuum-subtracted emission line.

    Parameters
    ----------
    spectrum : Spectrum
        The SAME unmodified spectrum passed to prepare_line, including its
        wavelength grid, flux, variance and mask. Do not reuse preparation
        from another spectrum or mutate the spectrum between stages.
    prepared : dict
        Output of prepare_line for this spectrum and line. It supplies the
        continuum, fit mask, initial parameters and parameter bounds.
    max_evaluations : int
        Positive maximum number of optimizer function evaluations.

    Returns
    -------
    dict
        Fitted parameters, conditional covariance, formal parameter errors,
        integrated Gaussian flux, observed FWHM, selected-channel model and
        residual arrays, numerical flags, bounds, units and preparation record.
        Covariance rows and columns follow parameter_order, namely amplitude,
        center, sigma. All results are restored to original flux/wavelength
        units. status='converged' means numerical convergence only.

    Raises
    ------
    ValueError
        If inputs are incompatible or insufficient valid channels remain.
    RuntimeError
        If the optimizer fails to converge or returns nonfinite parameters.

    Notes
    -----
    Like the notebook, this uses curve_fit with sigma=sqrt(variance),
    absolute_sigma=True, a Gaussian with nonnegative amplitude, and a fixed
    previously subtracted continuum. Valid negative and zero flux channels
    remain in the fit. A dimensionless internal coordinate and common flux/
    error scaling improve numerical conditioning; no observed data are
    interpolated or shifted, and the covariance is transformed back.

    Formal errors are conditional on the continuum and selected windows and
    assume independent channels. They do not include continuum sensitivity,
    calibration errors or the between-line statistical estimator in the paper.
    Those contributions belong to uncertainties.py. Bound-limited and singular
    fits are flagged, not silently accepted as reliable measurements.
    """
    if type(max_evaluations) is not int or max_evaluations < 1:
        raise ValueError('max_evaluations must be a positive integer.')
    if prepared.get('status') != 'prepared':
        raise ValueError('Successful line preparation is required.')
    for key in ('galaxy_id', 'region_id', 'wavelength_unit', 'wavelength_convention'):
        if prepared.get(key) != getattr(spectrum, key):
            raise ValueError(f'Preparation and spectrum disagree on {key}.')
    valid = build_valid_mask(spectrum)
    mask = np.asarray(prepared['fit_mask'])
    continuum = np.asarray(prepared['continuum'], dtype=float)
    if mask.dtype != bool or mask.shape != spectrum.flux.shape or continuum.shape != mask.shape:
        raise ValueError('Prepared mask and continuum must match the spectrum.')
    selected = mask & valid & np.isfinite(continuum)
    minimum = max(4, prepared['options']['min_fit_channels'])
    if np.count_nonzero(selected) < minimum:
        raise ValueError('Insufficient valid channels for a three-parameter Gaussian.')
    initial, lower, upper = initialize_parameters(prepared)
    wave = spectrum.wavelength[selected].copy()
    flux = (spectrum.flux - continuum)[selected].copy()
    error = np.sqrt(spectrum.variance[selected])
    if not np.allclose(flux, np.asarray(prepared['flux_without_continuum'])[selected],
                       rtol=1e-12, atol=0):
        raise ValueError('Spectrum flux changed after preparation.')
    reference = float(prepared['expected_wavelength'])
    width_scale = float(initial[2])
    flux_scale = max(float(np.max(np.abs(flux))), float(np.median(error)))
    scales = np.array([flux_scale, width_scale, width_scale])
    offsets = np.array([0., reference, 0.])
    x = (wave-reference) / width_scale
    p0 = (initial-offsets) / scales
    scaled_bounds = ((lower-offsets)/scales, (upper-offsets)/scales)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', OptimizeWarning)
        popt, pcov = curve_fit(
            gaussian, x, flux/flux_scale, p0=p0,
            sigma=error/flux_scale, absolute_sigma=True,
            bounds=scaled_bounds, maxfev=max_evaluations,
        )
    parameters = offsets + scales*popt
    if not np.all(np.isfinite(parameters)):
        raise RuntimeError('Optimizer returned nonfinite parameters.')
    covariance = pcov * scales[:, None] * scales[None, :]
    covariance_valid = bool(np.all(np.isfinite(pcov)))
    if covariance_valid:
        covariance_valid = bool(np.all(np.linalg.eigvalsh((pcov+pcov.T)/2) > 0))
    covariance_valid = covariance_valid and bool(np.all(np.isfinite(covariance)))
    formal_errors = np.sqrt(np.diag(covariance)) if covariance_valid else np.full(3, np.nan)
    # Check proximity in dimensionless coordinates; flag rather than reject.
    at_lower = np.isclose(popt, scaled_bounds[0], rtol=1e-5, atol=1e-7)
    at_upper = np.isclose(popt, scaled_bounds[1], rtol=1e-5, atol=1e-7)
    model = gaussian(wave, *parameters)
    diagnostics = compute_fit_diagnostics(flux, model, error)
    amplitude, center, sigma = map(float, parameters)
    return {
        'status': 'converged', 'line_id': prepared['line_id'],
        'galaxy_id': spectrum.galaxy_id, 'region_id': spectrum.region_id,
        'arm': spectrum.arm, 'z_initial': spectrum.z_initial,
        'wavelength_unit': spectrum.wavelength_unit, 'flux_unit': spectrum.flux_unit,
        'wavelength_convention': spectrum.wavelength_convention,
        'spectral_frame': spectrum.spectral_frame,
        'parameter_order': PARAMETER_ORDER,
        'parameters': dict(zip(PARAMETER_ORDER, map(float, parameters))),
        'parameter_errors_fit': dict(zip(PARAMETER_ORDER, map(float, formal_errors))),
        'covariance_fit': covariance,
        'integrated_flux': float(amplitude*sigma*np.sqrt(2*np.pi)),
        'integrated_flux_unit_factors': (spectrum.flux_unit, spectrum.wavelength_unit),
        'fwhm': float(sigma*2*np.sqrt(2*np.log(2))),
        'width_definition': 'observed; no instrumental deconvolution',
        'fit_mask': selected.copy(), 'wavelength_fit': wave, 'flux_fit': flux,
        'error_fit': error, 'model_fit': model,
        'model_with_continuum_fit': model+continuum[selected],
        'diagnostics': diagnostics,
        'flags': {'covariance_invalid': not covariance_valid,
                  'parameter_at_bound': bool(np.any(at_lower | at_upper)),
                  'fit_window_clipped_by_neighbor': prepared['fit_clipped_by_neighbor']},
        'bound_parameters': [name for name, flag in zip(PARAMETER_ORDER, at_lower | at_upper) if flag],
        'optimizer_warnings': [str(w.message) for w in caught],
        'uncertainty_scope': 'fit only; fixed continuum and windows; independent channel errors',
        'preparation': deepcopy(prepared),
    }
