"""Separate fitting errors, continuum sensitivity and between-line statistics.

Continuum-window dispersion is a sensitivity diagnostic, not automatically
an independent Gaussian error. No quadrature total is assigned by this module.
"""

from copy import deepcopy

import numpy as np
from astropy import units as u

from .data_models import Spectrum
from .fitting import fit_emission_line
from .preparation import build_line_windows, fit_continuum, C_KM_S

ORDER = ('amplitude', 'center', 'sigma')


def parameter_errors_from_covariance(covariance: np.ndarray) -> dict:
    """Return formal standard errors in amplitude, center, sigma order.

    Require a finite symmetric positive-semidefinite 3x3 covariance matrix.
    Validate positive semidefiniteness on a dimensionless correlation scale
    so different flux/wavelength units do not dominate the check. Raise
    ValueError for invalid matrices; do not take absolute values of variances.
    """
    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (3, 3) or not np.all(np.isfinite(cov)):
        raise ValueError('A finite 3x3 covariance matrix is required.')
    diagonal = np.diag(cov)
    if np.any(diagonal < 0):
        raise ValueError('Covariance diagonal must be nonnegative.')
    errors = np.sqrt(diagonal)
    for i in np.flatnonzero(errors == 0):
        if np.any(cov[i] != 0) or np.any(cov[:, i] != 0):
            raise ValueError('A zero variance requires zero associated covariances.')
    scale = np.where(errors > 0, errors, 1.)
    correlation = cov / scale[:, None] / scale[None, :]
    if not np.allclose(correlation, correlation.T, rtol=1e-10, atol=1e-12):
        raise ValueError('Covariance matrix must be symmetric.')
    if np.min(np.linalg.eigvalsh((correlation + correlation.T)/2)) < -1e-10:
        raise ValueError('Covariance matrix must be positive semidefinite.')
    return dict(zip(ORDER, map(float, errors)))


def propagate_line_flux_error(amplitude: float, sigma: float, covariance: np.ndarray) -> float:
    """Propagate F=A*sigma*sqrt(2*pi), INCLUDING amplitude-width covariance.

    covariance follows (amplitude, center, sigma). The returned error has
    units of spectral flux times wavelength. This is a first-order formal
    error conditional on the continuum; it excludes continuum sensitivity.
    """
    parameter_errors_from_covariance(covariance)
    if not np.all(np.isfinite([amplitude, sigma])) or amplitude < 0 or sigma <= 0:
        raise ValueError('Require finite nonnegative amplitude and positive sigma.')
    cov = np.asarray(covariance, dtype=float)
    gradient = np.sqrt(2*np.pi)*np.array([sigma, 0., amplitude])
    variance = float(gradient @ cov @ gradient)
    # Only clip round-off at a vanishing variance after the PSD check.
    tolerance = 1e-12 * float(np.abs(gradient) @ np.abs(cov) @ np.abs(gradient))
    if variance < -tolerance:
        raise ValueError('Propagated flux variance is negative.')
    return float(np.sqrt(max(variance, 0.)))


def summarize_fit_uncertainties(fit: dict) -> dict:
    """Return parameter, integrated-flux and observed-FWHM formal errors.

    Bound-limited fits retain their local formal errors but are explicitly
    flagged: symmetric errors near bounds are not calibrated confidence
    intervals. Invalid covariance returns NaNs and a reason, never zeros.
    The input result is not modified. No redshift conversion is done here.
    """
    output = {'scope': 'fit only; fixed continuum/windows; independent channels',
              'parameter_at_bound': fit['flags']['parameter_at_bound'],
              'wavelength_unit': fit['wavelength_unit'],
              'integrated_flux_unit_factors': fit['integrated_flux_unit_factors']}
    try:
        if tuple(fit['parameter_order']) != ORDER:
            raise ValueError('Unexpected covariance parameter order.')
        if fit['flags']['covariance_invalid']:
            raise ValueError('Fitter marked the covariance as invalid.')
        errors = parameter_errors_from_covariance(fit['covariance_fit'])
        flux_error = propagate_line_flux_error(fit['parameters']['amplitude'],
                                               fit['parameters']['sigma'], fit['covariance_fit'])
    except ValueError as exc:
        return {**output, 'status': 'unavailable', 'reason': str(exc),
                'parameter_errors_fit': dict.fromkeys(ORDER, np.nan),
                'integrated_flux_error_fit': np.nan, 'fwhm_error_fit': np.nan}
    return {**output, 'status': 'available', 'parameter_errors_fit': errors,
            'integrated_flux_error_fit': flux_error,
            'fwhm_error_fit': float(2*np.sqrt(2*np.log(2))*errors['sigma'])}


def evaluate_continuum_sensitivity(
    spectrum: Spectrum, prepared: dict, catalog: dict, *, factors=(0.85, 1.0, 1.15),
) -> dict:
    """Automatically vary continuum bands and repeat the line fit.

    Scale inner and outer continuum distances by each factor around the
    automatic baseline. Keep the original line-fitting mask and parameter
    bounds FIXED to isolate sensitivity to continuum choices. Bands exclude
    catalogued neighbours (including disabled doublets). Factors must keep
    the continuum outside the original +/-4-sigma fit window.

    Return per-variant results/failures, masks, bounds, sample standard
    deviations (ddof=1) and ranges for center, sigma, amplitude, integrated
    flux and FWHM. Duplicate masks are reported but counted only once.
    Failed or bound-limited fits do not enter the dispersion. Fewer than
    two distinct successful masks give unavailable dispersion (NaN).

    Partial results are explicitly labelled. This is not a calibrated 1-sigma
    error, and is not automatically added to formal errors. The perturbation
    factors are reproducible implementation choices, not values in the paper.
    """
    factors = np.asarray(factors, dtype=float)
    if factors.ndim != 1 or factors.size < 2 or not np.all(np.isfinite(factors)):
        raise ValueError('Provide at least two finite continuum factors.')
    if np.any(factors*5 <= 4) or len(np.unique(factors)) != factors.size:
        raise ValueError('Factors must be distinct and keep the continuum beyond 4 sigma.')
    if catalog['catalog_id'] != prepared['catalog_id']:
        raise ValueError('Use the same catalogue as in preparation.')
    reference = prepared['window_reference']
    options = prepared['options']
    expected = prepared['expected_wavelength']
    conversion = u.Unit(catalog['wavelength_unit']).to(u.Unit(spectrum.wavelength_unit))
    neighbors = []
    for key, line in catalog['lines'].items():
        if key != prepared['line_id']:
            values = line.get('components', [line.get('rest_wavelength')])
            neighbors.extend(x*conversion*(1+spectrum.z_initial) for x in values if x is not None)
    guard = max(3*expected*options['instrumental_sigma_kms']/C_KM_S, 4*reference['sigma'])
    variants, seen, measurements = [], set(), []
    names = (*ORDER, 'integrated_flux', 'fwhm')
    for factor in factors:
        row = {'factor': float(factor)}
        try:
            windows = build_line_windows(
                spectrum, expected, reference['center'], reference['sigma'],
                np.array(neighbors), guard, continuum_inner_sigma=5*factor,
                continuum_outer_sigma=options['continuum_outer_sigma']*factor)
            mask = windows['continuum_mask']
            row.update(continuum_mask=mask.copy(), continuum_bounds=windows['continuum_bounds'])
            if any(windows[k].sum() < options['min_continuum_channels_per_side']
                   for k in ('blue_continuum_mask', 'red_continuum_mask')):
                raise ValueError('Insufficient continuum channels on one or both sides.')
            signature = mask.tobytes()
            if signature in seen:
                variants.append({**row, 'status': 'duplicate_mask'})
                continue
            seen.add(signature)
            continuum = fit_continuum(spectrum, mask, expected)
            alternative = deepcopy(prepared)
            alternative.update(continuum=continuum['continuum'], continuum_fit=continuum,
                               flux_without_continuum=spectrum.flux-continuum['continuum'],
                               continuum_mask=mask, continuum_bounds=windows['continuum_bounds'],
                               blue_continuum_mask=windows['blue_continuum_mask'],
                               red_continuum_mask=windows['red_continuum_mask'])
            fit = fit_emission_line(spectrum, alternative)
            if fit['flags']['covariance_invalid'] or fit['flags']['parameter_at_bound']:
                raise ValueError('Variant has invalid covariance or a bound-limited fit.')
            values = {**fit['parameters'], 'integrated_flux': fit['integrated_flux'], 'fwhm': fit['fwhm']}
            row.update(status='ok', measurements=values)
            measurements.append([values[name] for name in names])
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            row.update(status='failed', reason=str(exc))
        variants.append(row)
    n = len(measurements)
    failure = any(row['status'] == 'failed' for row in variants)
    std = np.std(measurements, axis=0, ddof=1) if n >= 2 else np.full(len(names), np.nan)
    span = np.ptp(measurements, axis=0) if n >= 2 else np.full(len(names), np.nan)
    return {'status': 'unavailable' if n < 2 else ('partial' if failure else 'available'),
            'scope': 'continuum-window sensitivity; not a calibrated independent error',
            'n_distinct_successful_masks': n, 'variants': variants,
            'dispersion': dict(zip(names, map(float, std))),
            'range': dict(zip(names, map(float, span))), 'ddof': 1,
            'wavelength_unit': spectrum.wavelength_unit,
            'line_id': prepared['line_id'], 'region_id': spectrum.region_id}


def estimate_between_line_scatter(records: list[dict], *, ddof: int = 1) -> dict:
    """Estimate pooled within-galaxy delta-z scatter and its sample error.

    records contains galaxy_id, line_id and finite delta_z for one measurement
    per galaxy/line. Each delta_z must already compare the SAME line between
    the central and outer annuli. Pass AGN/non-AGN and red-only samples in
    separate calls; this function does not infer classes or select detections.

    Subtract each galaxy's unweighted mean delta_z, pool deviations from
    galaxies with >=2 lines, and divide pooled standard deviation by sqrt of
    the number of contributing GALAXIES, not the number of lines. This follows
    the statistical construction described in Section 3; ddof (0 or 1) is an
    explicit implementation choice because that detail is not specified there.
    Galaxies with only one line are reported as excluded. At least two eligible
    galaxies are required here for a sample-level error; otherwise return NaN.

    This sample-level statistic is NOT an individual-line fitting error and
    is not assigned to every line. Different line counts weight the pooled
    scatter differently. Correlations between lines remain a limitation.
    """
    if type(ddof) is not int or ddof not in (0, 1):
        raise ValueError('ddof must be 0 or 1.')
    groups, seen = {}, set()
    for row in records:
        galaxy, line, value = row['galaxy_id'], row['line_id'], row['delta_z']
        if galaxy is None or line is None or not np.isfinite(value):
            raise ValueError('Galaxy/line IDs and finite delta_z are required.')
        key = (galaxy, line)
        if key in seen:
            raise ValueError('Duplicate galaxy/line pair; do not count repeats as independent lines.')
        seen.add(key)
        groups.setdefault(galaxy, []).append(float(value))
    residuals, summary, excluded = [], [], []
    for galaxy, values in groups.items():
        if len(values) < 2:
            excluded.append(galaxy)
            continue
        mean = float(np.mean(values))
        residuals.extend(np.asarray(values)-mean)
        summary.append({'galaxy_id': galaxy, 'n_lines': len(values),
                        'mean_delta_z': mean, 'std_delta_z': float(np.std(values, ddof=ddof))})
    n = len(summary)
    scatter = float(np.std(residuals, ddof=ddof)) if n >= 2 else np.nan
    return {'status': 'available' if n >= 2 else 'insufficient_galaxies',
            'n_galaxies': n, 'n_lines': len(residuals), 'ddof': ddof,
            'pooled_within_galaxy_scatter': scatter,
            'sample_error': float(scatter/np.sqrt(n)) if n >= 2 else np.nan,
            'galaxies': summary, 'excluded_single_line_galaxies': excluded,
            'scope': 'sample-level between-line statistic; not individual fit uncertainty'}
