"""Convert fitted wavelengths to redshifts and central-minus-outer differences.

No gravitational interpretation is assigned automatically. Formal fit errors
remain separate from continuum sensitivity and sample-level statistics.
"""

import numpy as np
from astropy import units as u

C_KM_S = 299792.458


def compute_redshift(fit: dict, catalog: dict) -> dict:
    """Compute z=lambda_fit/lambda_rest-1 and its formal fitting error.

    Parameters
    ----------
    fit : dict
        Result of fit_emission_line, including parameters, covariance flags,
        units, galaxy/region identifiers and preparation provenance.
    catalog : dict
        The same validated line catalogue used for preparation.

    Returns
    -------
    dict
        Dimensionless z and z_error_fit, along with the measured wavelength,
        adopted rest wavelength in the same unit, identifiers and fit flags.
        Invalid fitting covariance gives an unavailable error (NaN), while
        retaining the fitted redshift. Bound flags are retained for evaluation.

    Notes
    -----
    Rest wavelengths are treated as fixed references. This propagation does
    not include atomic-reference, wavelength-calibration or continuum errors.
    No air/vacuum conversion is performed: conventions must already agree.
    Unit conversion is explicit. The original spectral reference frame is
    preserved, not inferred. The initial catalogue redshift is not subtracted.
    """
    if fit.get('status') != 'converged':
        raise ValueError('A converged line fit is required.')
    line_id = fit['line_id']
    line = catalog['lines'][line_id]
    if not line['enabled'] or line['rest_wavelength'] is None:
        raise ValueError('An enabled line with a defined rest wavelength is required.')
    preparation = fit['preparation']
    if (catalog['catalog_id'] != preparation['catalog_id']
            or line != preparation['catalog_line']):
        raise ValueError('The catalogue or line definition changed after preparation.')
    if (fit['wavelength_convention'] not in ('air', 'vacuum')
            or fit['wavelength_convention'] != catalog['wavelength_convention']):
        raise ValueError('Known matching air/vacuum conventions are required.')
    rest = float((line['rest_wavelength']*u.Unit(catalog['wavelength_unit'])).to_value(
        u.Unit(fit['wavelength_unit'])))
    center = float(fit['parameters']['center'])
    if not np.all(np.isfinite([rest, center])) or min(rest, center) <= 0:
        raise ValueError('Rest and fitted wavelengths must be finite and positive.')
    error = float(fit['parameter_errors_fit']['center'])
    available = not fit['flags']['covariance_invalid'] and np.isfinite(error) and error >= 0
    return {
        'galaxy_id': fit['galaxy_id'], 'region_id': fit['region_id'], 'line_id': line_id,
        'catalog_id': catalog['catalog_id'], 'arm': fit['arm'],
        'wavelength_convention': fit['wavelength_convention'],
        'spectral_frame': fit['spectral_frame'], 'wavelength_unit': fit['wavelength_unit'],
        'lambda_fit': center, 'lambda_rest': rest,
        'lambda_error_fit': error if available else np.nan,
        'z': float(center/rest-1), 'z_error_fit': float(error/rest) if available else np.nan,
        'z_initial': fit['z_initial'], 'fit_flags': dict(fit['flags']),
        'error_status': 'available' if available else 'unavailable',
        'uncertainty_scope': 'formal fit error; fixed rest wavelength and continuum',
    }


def _difference_error(error_a: float, error_b: float, covariance: float) -> float:
    """Propagate var(a-b), requiring a physically admissible covariance."""
    if not np.isfinite(covariance):
        raise ValueError('Covariance must be finite.')
    if not np.isfinite(error_a) or not np.isfinite(error_b):
        return np.nan
    if error_a < 0 or error_b < 0:
        raise ValueError('Standard errors must be nonnegative.')
    limit = error_a*error_b
    if abs(covariance) > limit and not np.isclose(abs(covariance), limit, rtol=1e-12, atol=0):
        raise ValueError('Covariance exceeds the product of standard errors.')
    variance = error_a**2 + error_b**2 - 2*covariance
    return float(np.sqrt(max(variance, 0.)))


def compute_delta_redshift(
    central: dict, outer: dict, *, covariance_z: float = 0.0,
) -> dict:
    """Compare the same line in central BIN_MASK=1 and outer BIN_MASK=4.

    Parameters
    ----------
    central, outer : dict
        Results from compute_redshift for the same galaxy and line.
    covariance_z : float
        Covariance between central and outer redshift FIT errors, in z^2.
        Zero reproduces the independent-error quadrature in the notebook.
        A nonzero value must be supplied from a justified covariance model.

    Returns
    -------
    dict
        delta_z=z_central-z_outer, propagated delta_z_error_fit, paired
        reference/identity information and covariance assumption. Negative
        and zero differences are retained. Unknown but matching frame labels
        are explicitly flagged, not presented as verified frame agreement.

    Notes
    -----
    Both region IDs are enforced as agreed for this SAMI implementation.
    Different wavelength units are allowed if the adopted physical rest
    wavelengths agree. No division by (1+z_initial) is applied to delta_z.
    This observable is not automatically labelled gravitational redshift.
    """
    for key in ('galaxy_id', 'line_id', 'catalog_id', 'wavelength_convention', 'spectral_frame'):
        if central[key] != outer[key]:
            raise ValueError(f'Central and outer measurements disagree on {key}.')
    if central['galaxy_id'] is None:
        raise ValueError('A galaxy identifier is required to pair measurements.')
    if central['region_id'] != 1 or outer['region_id'] != 4:
        raise ValueError('Expected central region 1 and outer region 4.')
    rest_outer = (outer['lambda_rest']*u.Unit(outer['wavelength_unit'])).to_value(
        u.Unit(central['wavelength_unit']))
    if not np.isclose(central['lambda_rest'], rest_outer, rtol=1e-12, atol=0):
        raise ValueError('Central and outer measurements use different rest wavelengths.')
    if not np.all(np.isfinite([central['z'], outer['z']])):
        raise ValueError('Both measured redshifts must be finite.')
    error = _difference_error(central['z_error_fit'], outer['z_error_fit'], covariance_z)
    return {
        'galaxy_id': central['galaxy_id'], 'line_id': central['line_id'],
        'catalog_id': central['catalog_id'], 'central_region_id': 1, 'outer_region_id': 4,
        'z_central': central['z'], 'z_outer': outer['z'],
        'z_central_error_fit': central['z_error_fit'], 'z_outer_error_fit': outer['z_error_fit'],
        'delta_z': float(central['z']-outer['z']), 'delta_z_error_fit': error,
        'covariance_z': float(covariance_z),
        'error_assumption': 'independent central/outer fit errors' if covariance_z == 0 else 'supplied covariance',
        'error_status': 'available' if np.isfinite(error) else 'unavailable',
        'spectral_frame': central['spectral_frame'],
        'spectral_frame_verified': central['spectral_frame'] != 'unknown',
        'wavelength_convention': central['wavelength_convention'],
        'central_fit_flags': dict(central['fit_flags']), 'outer_fit_flags': dict(outer['fit_flags']),
        'quantity': 'observed central-minus-outer redshift difference',
    }


def compute_delta_velocity(
    delta_z: float, delta_z_error: float, z_reference: float, *,
    z_reference_error: float = 0.0, covariance_delta_z_reference: float = 0.0,
) -> dict:
    """Convert delta_z to approximate rest-frame velocity in km/s.

    Use delta_v=c*delta_z/(1+z_reference), as in the notebook. This is the
    small-relative-shift approximation, not an exact relativistic velocity.
    The reference is normally the shared initial galaxy redshift, explicitly
    supplied by the caller. It is NOT the fitted central or outer redshift
    unless the caller intentionally selects that reference.

    Errors use first-order propagation in delta_z and z_reference, including
    optional covariance (in redshift squared). Defaults treat the reference
    as fixed, matching the notebook. A NaN input delta_z_error propagates as
    unavailable uncertainty. Negative and zero velocity differences are kept.
    """
    if not np.all(np.isfinite([delta_z, z_reference, z_reference_error,
                               covariance_delta_z_reference])):
        raise ValueError('Velocity inputs and reference error/covariance must be finite.')
    if z_reference <= -1 or z_reference_error < 0:
        raise ValueError('Require z_reference > -1 and nonnegative reference error.')
    a = C_KM_S/(1+z_reference)
    b = -C_KM_S*delta_z/(1+z_reference)**2
    if np.isfinite(delta_z_error):
        # Validate the covariance before using the full Jacobian expression.
        _difference_error(delta_z_error, z_reference_error, covariance_delta_z_reference)
        variance = (a*delta_z_error)**2 + (b*z_reference_error)**2 + 2*a*b*covariance_delta_z_reference
        error = float(np.sqrt(max(variance, 0.)))
    else:
        error = np.nan
    return {'delta_v_km_s': float(a*delta_z), 'delta_v_error_km_s': error,
            'z_reference': float(z_reference), 'z_reference_error': float(z_reference_error),
            'covariance_delta_z_reference': float(covariance_delta_z_reference),
            'approximation': 'c * delta_z / (1 + z_reference)',
            'error_status': 'available' if np.isfinite(error) else 'unavailable'}
