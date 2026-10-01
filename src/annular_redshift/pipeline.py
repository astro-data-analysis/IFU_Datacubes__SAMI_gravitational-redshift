"""Coordinate automated SAMI annular line measurements and retain failures."""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import load_line_catalog, _UniqueKeySafeLoader
from .loaders.sami import load_sami_annular_cube
from .extraction import extract_annular_spectrum
from .curation import curate_spectrum
from .preparation import prepare_line
from .fitting import fit_emission_line
from .uncertainties import summarize_fit_uncertainties, evaluate_continuum_sensitivity
from .redshift import compute_redshift, compute_delta_redshift, compute_delta_velocity
from .results import build_result_row, assemble_results_table


EXPECTED_FAILURES = (ValueError, RuntimeError, KeyError, OSError, np.linalg.LinAlgError)


def _load_analysis_config(path: str | Path) -> dict:
    """Validate the currently supported SAMI analysis configuration."""
    with Path(path).expanduser().open(encoding='utf-8') as stream:
        try:
            config = yaml.load(stream, Loader=_UniqueKeySafeLoader)
        except yaml.YAMLError as exc:
            raise ValueError(f'Invalid analysis YAML: {exc}') from exc
    if not isinstance(config, dict):
        raise ValueError('Analysis configuration must be a mapping.')
    expected = {'schema_version', 'config_id', 'central_region', 'outer_region',
                'instrumental_sigma_kms', 'instrumental_resolution_source', 'line_ids',
                'extraction', 'preparation', 'fitting', 'continuum_sensitivity', 'covariance_z'}
    if set(config) != expected:
        raise ValueError('Analysis configuration has missing or unsupported keys.')
    if type(config['schema_version']) is not int or config['schema_version'] != 1:
        raise ValueError('Analysis schema_version must be integer 1.')
    if not isinstance(config['config_id'], str) or not config['config_id'].strip():
        raise ValueError('config_id must be a nonempty string.')
    if (type(config['central_region']) is not int or config['central_region'] != 1
            or type(config['outer_region']) is not int or config['outer_region'] != 4):
        raise ValueError('This implementation uses central=1 and outer=4.')
    sections = {
        'instrumental_sigma_kms': {'blue', 'red'},
        'extraction': {'rtol', 'atol'},
        'preparation': {'search_half_width_kms', 'initial_width_factor', 'max_iterations',
                        'convergence_fraction', 'min_continuum_channels_per_side', 'min_fit_channels'},
        'fitting': {'max_evaluations'},
        'continuum_sensitivity': {'enabled', 'factors'},
    }
    for key, fields in sections.items():
        if not isinstance(config[key], dict) or set(config[key]) != fields:
            raise ValueError(f'Invalid fields in analysis section {key}.')
    for key in ('instrumental_sigma_kms', 'extraction', 'preparation', 'fitting'):
        for name, value in config[key].items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f'{key}.{name} must be finite and numeric.')
            if value < 0 or (key != 'extraction' and value == 0):
                raise ValueError(f'{key}.{name} has an invalid sign.')
    for section, name, minimum in [('preparation', 'max_iterations', 1),
                                  ('preparation', 'min_fit_channels', 4),
                                  ('preparation', 'min_continuum_channels_per_side', 2),
                                  ('fitting', 'max_evaluations', 1)]:
        value = config[section][name]
        if type(value) is not int or value < minimum:
            raise ValueError(f'{section}.{name} must be an integer >= {minimum}.')
    sensitivity = config['continuum_sensitivity']
    if type(sensitivity['enabled']) is not bool:
        raise ValueError('continuum_sensitivity.enabled must be boolean.')
    factors = sensitivity['factors']
    if (not isinstance(factors, list) or len(factors) < 2
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in factors)
            or not np.all(np.isfinite(factors)) or min(factors) <= .8
            or len(set(factors)) != len(factors)):
        raise ValueError('Continuum factors must be distinct finite numbers > 0.8.')
    ids = config['line_ids']
    if ids is not None and (not isinstance(ids, list) or not ids
                           or any(not isinstance(x, str) or not x for x in ids)
                           or len(ids) != len(set(ids))):
        raise ValueError('line_ids must be null or a nonempty list of unique strings.')
    if type(config['covariance_z']) not in (int, float) or config['covariance_z'] != 0:
        raise ValueError('The current automatic pipeline supports covariance_z=0 only.')
    return config


def measure_cube(
    path: str | Path, *, line_catalog_path: str | Path,
    analysis_config_path: str | Path, arm: str | None = None,
) -> pd.DataFrame:
    """Measure enabled lines in both annuli of one SAMI spectral-arm cube.

    Parameters
    ----------
    path : str or Path
        SAMI annular FITS or FITS.gz file.
    line_catalog_path, analysis_config_path : str or Path
        Explicit paths to lines.yaml and sami_annular.yaml. No dependence on
        the working directory or an unpublished package-data installation is
        hidden in this API; relative paths follow normal Python semantics.
    arm : {'blue', 'red'}, optional
        Explicit arm when the filename has been renamed (for example on web
        upload). Must agree with a recognized standard SAMI filename.

    Returns
    -------
    pandas.DataFrame
        One row per galaxy/region/selected line. Comparisons appear once,
        on central rows. The table's attrs contain configuration snapshots,
        input SHA256, catalogue/analysis SHA256 hashes, stage details and fit
        arrays. DataFrame attrs are in-memory metadata; CSV alone does not
        preserve them. Export/provenance persistence will need an explicit
        format when that stage is implemented.

    Notes
    -----
    Global configuration and arm-routing mistakes raise errors before fitting.
    Data loading, extraction and per-line scientific failures become rows with
    stage/reason. Successful individual fits survive a failed comparison.
    Unknown galaxy identifiers use 'source:<absolute path>' and are flagged
    in attrs; they are never guessed as physical galaxy IDs.

    Enabled lines are routed by catalogue preferred_arm; out-of-coverage or
    weak lines are attempted and produce failure rows as appropriate. Disabled
    entries remain listed in provenance but are not measured. No manual review
    or cut on delta-z sign/significance is applied. Fit quality flags remain
    descriptive; numerical convergence does not certify unimodality.
    """
    path = Path(path).expanduser()
    config = _load_analysis_config(analysis_config_path)
    catalog = load_line_catalog(line_catalog_path)
    if arm is not None and arm not in ('blue', 'red'):
        raise ValueError('arm must be blue, red or None.')
    requested = config['line_ids']
    ids = [key for key, line in catalog['lines'].items() if line['enabled']] if requested is None else requested
    for key in ids:
        if key not in catalog['lines'] or not catalog['lines'][key]['enabled']:
            raise ValueError(f'Requested line {key!r} is absent or disabled.')
    stem = path.name.lower().removesuffix('.gz').removesuffix('.fits')
    inferred = 'blue' if stem.endswith('_annular_blue') else 'red' if stem.endswith('_annular_red') else None
    if arm is not None and inferred is not None and arm != inferred:
        raise ValueError('Explicit arm disagrees with the standard filename.')
    selected_arm = arm or inferred
    if selected_arm is None:
        raise ValueError('Cannot identify the spectral arm; provide arm explicitly.')
    ids = [key for key in ids if catalog['lines'][key]['preferred_arm'] == selected_arm]
    regions = (config['central_region'], config['outer_region'])
    details, rows = {}, []
    provenance = {'analysis_config': deepcopy(config), 'line_catalog': deepcopy(catalog),
                  'source_path': str(path.resolve()), 'arm': selected_arm,
                  'analysis_sha256': sha256(Path(analysis_config_path).expanduser().read_bytes()).hexdigest(),
                  'catalog_sha256': sha256(Path(line_catalog_path).expanduser().read_bytes()).hexdigest(),
                  'selected_lines': ids,
                  'disabled_lines': [key for key, line in catalog['lines'].items() if not line['enabled']],
                  'details': details}
    source_id = 'source:'+str(path.resolve())
    try:
        digest = sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
        provenance['source_sha256'] = digest.hexdigest()
        cube = load_sami_annular_cube(path)
    except EXPECTED_FAILURES as exc:
        for key in ids:
            for region in regions:
                rows.append(build_result_row(galaxy_id=source_id, region_id=region, line_id=key,
                    catalog_id=catalog['catalog_id'], arm=selected_arm,
                    failure_stage='loading', failure_reason=str(exc)))
        table = assemble_results_table(rows)
        table.attrs.update(provenance)
        return table
    provenance['galaxy_id_fallback'] = cube.galaxy_id is None
    cube.galaxy_id = cube.galaxy_id or source_id
    cube.arm = selected_arm
    spectra, region_failures = {}, {}
    for region in regions:
        stage = 'extraction'
        try:
            spectrum = extract_annular_spectrum(cube, region, **config['extraction'])
            stage = 'curation'
            spectra[region] = curate_spectrum(spectrum)
        except EXPECTED_FAILURES as exc:
            region_failures[region] = (stage, str(exc))
    for key in ids:
        line_results = {}
        for region in regions:
            result = {}
            line_results[region] = result
            if region in region_failures:
                result['failure_stage'], result['failure_reason'] = region_failures[region]
                continue
            spectrum = spectra[region]
            stage = 'preparation'
            try:
                prepared = prepare_line(spectrum, catalog, key,
                    instrumental_sigma_kms=config['instrumental_sigma_kms'][selected_arm],
                    **config['preparation'])
                result['prepared'] = prepared
                stage = 'fitting'
                result['fit'] = fit_emission_line(spectrum, prepared, **config['fitting'])
                stage = 'fit_uncertainties'
                result['fit_uncertainties'] = summarize_fit_uncertainties(result['fit'])
                stage = 'redshift'
                result['redshift'] = compute_redshift(result['fit'], catalog)
            except EXPECTED_FAILURES as exc:
                result.update(failure_stage=stage, failure_reason=str(exc))
            if config['continuum_sensitivity']['enabled'] and 'redshift' in result:
                try:
                    result['continuum_sensitivity'] = evaluate_continuum_sensitivity(
                        spectrum, result['prepared'], catalog,
                        factors=config['continuum_sensitivity']['factors'])
                except EXPECTED_FAILURES as exc:
                    # Preserve the successful line fit if this optional diagnostic fails.
                    result['continuum_sensitivity'] = {
                        'status': 'unavailable', 'reason': str(exc), 'line_id': key,
                        'region_id': region, 'wavelength_unit': spectrum.wavelength_unit,
                        'n_distinct_successful_masks': 0,
                        'dispersion': dict.fromkeys(('center','amplitude','sigma','fwhm','integrated_flux'), np.nan)}
        comparison = velocity = None
        comparison_failure = None
        central, outer = (line_results[r] for r in regions)
        if 'redshift' in central and 'redshift' in outer:
            try:
                comparison = compute_delta_redshift(central['redshift'], outer['redshift'])
                velocity = compute_delta_velocity(comparison['delta_z'], comparison['delta_z_error_fit'], cube.z_initial)
            except EXPECTED_FAILURES as exc:
                comparison_failure = str(exc)
                comparison = velocity = None
        else:
            comparison_failure = 'Central and outer redshifts are both required; see individual failure rows.'
        for region in regions:
            result = line_results[region]
            kwargs = {name: result.get(name) for name in ('fit','redshift','fit_uncertainties',
                       'continuum_sensitivity','failure_stage','failure_reason')}
            rows.append(build_result_row(galaxy_id=cube.galaxy_id, region_id=region, line_id=key,
                catalog_id=catalog['catalog_id'], arm=selected_arm,
                comparison=comparison if region == 1 else None,
                velocity=velocity if region == 1 else None,
                comparison_failure_reason=comparison_failure if region == 1 else None, **kwargs))
        details[key] = {'regions': line_results, 'comparison': comparison, 'velocity': velocity}
    table = assemble_results_table(rows)
    table.attrs.update(provenance)
    return table


def measure_sample(
    paths, *, line_catalog_path: str | Path, analysis_config_path: str | Path,
) -> pd.DataFrame:
    """Run measure_cube for paths with standard blue/red SAMI filenames.

    Preserve input order and per-cube provenance in attrs['runs']. Reject
    duplicate galaxy/region/line rows rather than silently merging repeat
    observations. Supply blue and red products of each galaxy as separate
    paths. A flat single table is returned; no sample-level error is assigned
    automatically to individual rows. An empty iterable returns an empty schema.
    """
    rows, runs = [], []
    for path in paths:
        table = measure_cube(path, line_catalog_path=line_catalog_path,
                             analysis_config_path=analysis_config_path)
        rows.extend(table.to_dict(orient='records'))
        runs.append(table.attrs)
    result = assemble_results_table(rows)
    result.attrs['runs'] = runs
    return result
