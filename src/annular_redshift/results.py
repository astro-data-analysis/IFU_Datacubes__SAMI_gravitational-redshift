"""Assemble one flat results table with one row per galaxy, region and line."""

import json

import numpy as np
import pandas as pd


COLUMNS = [
    'galaxy_id', 'region_id', 'line_id', 'catalog_id', 'arm',
    'measurement_status', 'failure_stage', 'failure_reason',
    'wavelength_unit', 'flux_unit', 'wavelength_convention', 'spectral_frame',
    'z_initial', 'lambda_rest', 'lambda_fit', 'lambda_error_fit',
    'amplitude', 'amplitude_error_fit', 'sigma', 'sigma_error_fit',
    'fwhm', 'fwhm_error_fit', 'integrated_flux', 'integrated_flux_error_fit',
    'integrated_flux_unit_factors', 'z', 'z_error_fit', 'fit_error_status',
    'n_fit_channels', 'degrees_of_freedom', 'chi_square', 'reduced_chi_square',
    'covariance_invalid', 'parameter_at_bound', 'bound_parameters',
    'fit_window_clipped_by_neighbor', 'optimizer_warnings',
    'continuum_sensitivity_status', 'n_continuum_variants',
    'lambda_continuum_dispersion', 'amplitude_continuum_dispersion',
    'sigma_continuum_dispersion', 'fwhm_continuum_dispersion',
    'integrated_flux_continuum_dispersion',
    'reference_region_id', 'comparison_status', 'comparison_failure_reason',
    'z_reference_region', 'delta_z', 'delta_z_error_fit', 'covariance_z',
    'delta_v_km_s', 'delta_v_error_fit_km_s', 'velocity_z_reference',
    'velocity_z_reference_error', 'velocity_covariance_delta_z_reference',
    'comparison_frame_verified',
]


def _check_identity(result: dict, galaxy_id, region_id, line_id, name: str) -> None:
    """Reject mismatched auxiliary results rather than joining them silently."""
    for key, expected in [('galaxy_id', galaxy_id), ('region_id', region_id), ('line_id', line_id)]:
        if key in result and result[key] != expected:
            raise ValueError(f'{name} disagrees on {key}.')


def build_result_row(
    *, galaxy_id, region_id: int, line_id: str,
    fit: dict | None = None,
    redshift: dict | None = None,
    fit_uncertainties: dict | None = None,
    continuum_sensitivity: dict | None = None,
    comparison: dict | None = None,
    velocity: dict | None = None,
    failure_stage: str | None = None,
    failure_reason: str | None = None,
    comparison_failure_reason: str | None = None,
    catalog_id: str | None = None,
    arm: str | None = None,
) -> dict:
    """Build a flat row for one galaxy/region/line, retaining failures.

    Parameters
    ----------
    galaxy_id, region_id, line_id
        Explicit row identifiers. region_id is an original BIN_MASK label.
    fit, redshift, fit_uncertainties, continuum_sensitivity : dict, optional
        Outputs of fit_emission_line, compute_redshift,
        summarize_fit_uncertainties, evaluate_continuum_sensitivity.
        They must refer to this row's measurement. Optional inputs may be
        absent when a processing stage failed or has not been run.
    comparison, velocity : dict, optional
        Outputs of compute_delta_redshift and compute_delta_velocity for
        this galaxy/line. Store them ONLY on central region 1, with outer
        region 4 as reference. The outer row retains its own line measurement
        and leaves comparison values missing; it is not a second delta-z.
    failure_stage, failure_reason : str, optional
        Supply both to record a failed processing stage. Earlier successful
        measurements, if supplied, remain visible in the failed row.
    comparison_failure_reason : str, optional
        Explain why the central/outer comparison failed, without marking
        a successful individual line measurement as failed.
    catalog_id, arm : optional
        Metadata useful even when fitting failed. Checked against a fit
        if one is supplied.

    Returns
    -------
    dict
        A fixed-column flat row. Missing numbers use NaN; absent metadata
        use None. Lists of warnings/flags are JSON strings for CSV-friendly
        storage. Formal errors and continuum sensitivity remain separate.
        No error total, manual decision or significance-based quality class
        is inferred. Convergence is not a scientific reliability label.

    Notes
    -----
    Full models, covariance matrices and variant histories remain in the
    original function outputs, not cells of this compact measurement table.
    Sample-level between-line errors belong to a separate statistical result;
    they must not be replicated as individual-line errors in this table.
    This module does not write files or display graphs.
    """
    if galaxy_id is None or not isinstance(line_id, str) or not line_id:
        raise ValueError('Galaxy and line identifiers are required.')
    if isinstance(region_id, (bool, np.bool_)) or not isinstance(region_id, (int, np.integer)):
        raise ValueError('region_id must be an integer.')
    if bool(failure_stage) != bool(failure_reason):
        raise ValueError('Provide both failure_stage and failure_reason.')
    if fit is None and redshift is not None:
        raise ValueError('A redshift result requires its matching fit.')
    if fit is None and fit_uncertainties is not None:
        raise ValueError('Fit uncertainties require their matching fit.')
    if comparison is not None and (region_id != 1 or comparison_failure_reason):
        raise ValueError('A comparison belongs only on a central row without a comparison failure.')
    if comparison_failure_reason and region_id != 1:
        raise ValueError('Comparison failures belong on the central row.')
    if velocity is not None and comparison is None:
        raise ValueError('A velocity conversion requires a comparison.')
    row = dict.fromkeys(COLUMNS, np.nan)
    row.update(galaxy_id=galaxy_id, region_id=int(region_id), line_id=line_id,
               catalog_id=catalog_id, arm=arm, failure_stage=failure_stage,
               failure_reason=failure_reason,
               measurement_status='failed' if failure_stage else 'not_run',
               comparison_status='not_run' if region_id == 1 else 'not_applicable',
               comparison_failure_reason=comparison_failure_reason,
               fit_error_status='not_evaluated', continuum_sensitivity_status='not_evaluated',
               bound_parameters='[]', optimizer_warnings='[]')
    if fit is not None:
        _check_identity(fit, galaxy_id, region_id, line_id, 'Fit')
        for key, expected in [('catalog_id', fit['preparation']['catalog_id']), ('arm', fit['arm'])]:
            if row[key] is not None and row[key] != expected:
                raise ValueError(f'Supplied {key} disagrees with fit.')
            row[key] = expected
        for key in ('wavelength_unit', 'flux_unit', 'wavelength_convention', 'spectral_frame', 'z_initial'):
            row[key] = fit[key]
        if not failure_stage:
            row['measurement_status'] = fit['status']
        for name in ('amplitude', 'sigma'):
            row[name] = fit['parameters'][name]
            row[name+'_error_fit'] = fit['parameter_errors_fit'][name]
        row.update(lambda_fit=fit['parameters']['center'],
                   lambda_error_fit=fit['parameter_errors_fit']['center'],
                   fwhm=fit['fwhm'], integrated_flux=fit['integrated_flux'],
                   integrated_flux_unit_factors=json.dumps(fit['integrated_flux_unit_factors']),
                   fit_error_status='unavailable' if fit['flags']['covariance_invalid'] else 'fit_parameters_only',
                   bound_parameters=json.dumps(fit['bound_parameters']),
                   optimizer_warnings=json.dumps(fit['optimizer_warnings']))
        row.update({k: fit['flags'][k] for k in (
            'covariance_invalid', 'parameter_at_bound', 'fit_window_clipped_by_neighbor')})
        row['n_fit_channels'] = fit['diagnostics']['n_channels']
        for key in ('degrees_of_freedom', 'chi_square', 'reduced_chi_square'):
            row[key] = fit['diagnostics'][key]
    if redshift is not None:
        _check_identity(redshift, galaxy_id, region_id, line_id, 'Redshift')
        if redshift['catalog_id'] != row['catalog_id']:
            raise ValueError('Redshift catalogue disagrees with fit.')
        if redshift['wavelength_unit'] != row['wavelength_unit']:
            raise ValueError('Redshift wavelength unit disagrees with fit.')
        if not np.isclose(redshift['lambda_fit'], row['lambda_fit'], rtol=1e-12, atol=0):
            raise ValueError('Redshift was computed from a different fitted center.')
        for key in ('lambda_rest', 'z', 'z_error_fit'):
            row[key] = redshift[key]
    if fit_uncertainties is not None:
        if fit_uncertainties['wavelength_unit'] != row['wavelength_unit']:
            raise ValueError('Uncertainty wavelength unit disagrees with fit.')
        row['fit_error_status'] = fit_uncertainties['status']
        for key in ('fwhm_error_fit', 'integrated_flux_error_fit'):
            row[key] = fit_uncertainties[key]
    if continuum_sensitivity is not None:
        _check_identity(continuum_sensitivity, galaxy_id, region_id, line_id, 'Continuum sensitivity')
        if fit is not None and continuum_sensitivity['wavelength_unit'] != row['wavelength_unit']:
            raise ValueError('Continuum wavelength unit disagrees with fit.')
        row['continuum_sensitivity_status'] = continuum_sensitivity['status']
        row['n_continuum_variants'] = continuum_sensitivity['n_distinct_successful_masks']
        for name, column in [('center', 'lambda'), ('amplitude', 'amplitude'),
                             ('sigma', 'sigma'), ('fwhm', 'fwhm'), ('integrated_flux', 'integrated_flux')]:
            row[column+'_continuum_dispersion'] = continuum_sensitivity['dispersion'][name]
    if comparison_failure_reason:
        row['comparison_status'] = 'failed'
        row['reference_region_id'] = 4
    if comparison is not None:
        if redshift is None:
            raise ValueError('A central comparison requires the central redshift result.')
        _check_identity(comparison, galaxy_id, region_id, line_id, 'Comparison')
        if comparison['central_region_id'] != 1 or comparison['outer_region_id'] != 4:
            raise ValueError('Expected central 1 and outer 4 in the comparison.')
        if comparison['catalog_id'] != row['catalog_id']:
            raise ValueError('Comparison catalogue disagrees with fit.')
        if not np.isclose(comparison['z_central'], row['z'], rtol=1e-12, atol=1e-15):
            raise ValueError('Comparison has a different central redshift.')
        row.update(reference_region_id=4, comparison_status='computed',
                   z_reference_region=comparison['z_outer'],
                   comparison_frame_verified=comparison['spectral_frame_verified'])
        for key in ('delta_z', 'delta_z_error_fit', 'covariance_z'):
            row[key] = comparison[key]
    if velocity is not None:
        expected_velocity = 299792.458*row['delta_z']/(1+velocity['z_reference'])
        if not np.isclose(velocity['delta_v_km_s'], expected_velocity, rtol=1e-12, atol=1e-12):
            raise ValueError('Velocity conversion is inconsistent with delta_z.')
        row.update(delta_v_km_s=velocity['delta_v_km_s'],
                   delta_v_error_fit_km_s=velocity['delta_v_error_km_s'],
                   velocity_z_reference=velocity['z_reference'],
                   velocity_z_reference_error=velocity['z_reference_error'],
                   velocity_covariance_delta_z_reference=velocity['covariance_delta_z_reference'])
    return row


def assemble_results_table(rows: list[dict]) -> pd.DataFrame:
    """Build the single measurement table, rejecting duplicate identities.

    Each row must be produced by build_result_row. Preserve input order,
    failures and missing comparisons. Empty input returns the full schema.
    No selection by delta-z sign, significance or manual inspection is done.
    """
    for row in rows:
        if set(row) != set(COLUMNS):
            raise ValueError('Each row must have the build_result_row schema.')
    table = pd.DataFrame(rows, columns=COLUMNS)
    keys = ['galaxy_id', 'region_id', 'line_id']
    if table.duplicated(keys).any():
        raise ValueError('Duplicate galaxy/region/line rows; distinguish separate runs before combining.')
    return table
