"""Read and validate the versioned emission-line catalogue."""

import math
from pathlib import Path

from astropy import units as u
import yaml


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Use safe YAML parsing and reject duplicate mapping keys."""

    def construct_mapping(self, node, deep=False):
        # SafeLoader normally overwrites duplicate keys silently.
        mapping = super().construct_mapping(node, deep=deep)
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise ValueError(
                    f"Duplicate YAML key {key!r} on line {key_node.start_mark.line + 1}."
                )
            seen.add(key)
        return mapping


def _require_text(value, field):
    """Require a nonempty string without changing its stored value."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string.")


def _require_wavelength(value, field):
    """Reject booleans, strings and nonpositive or nonfinite wavelengths."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a numeric wavelength, not a boolean or string.")
    try:
        valid = math.isfinite(value) and value > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{field} must be finite and strictly positive.")


def load_line_catalog(path: str | Path) -> dict:
    """Read and validate a schema-version-1 line catalogue from YAML.

    Parameters
    ----------
    path : str or pathlib.Path
        Explicit path to lines.yaml. Relative paths are resolved against
        the current working directory, not the package installation folder.

    Returns
    -------
    dict
        The validated catalogue, preserving identifiers, wavelength values,
        notes, source references and both enabled and disabled entries.
        For example, catalog['lines']['Halpha']['rest_wavelength'] gives
        the adopted Halpha rest wavelength in catalog['wavelength_unit'].
        No default values are inserted and no numerical values are changed.

    Raises
    ------
    OSError
        If the file cannot be read.
    ValueError
        If YAML is malformed, contains duplicate keys, has an unsupported
        schema version, or fails catalogue validation. Error messages name
        the relevant field where possible.

    Notes
    -----
    Safe YAML parsing is used; Python objects cannot be constructed from
    YAML tags. Duplicate keys are rejected rather than silently overwritten.
    Wavelength units must be convertible to length and the convention must
    be 'air' or 'vacuum'. Their agreement with a particular Spectrum will
    be checked during spectral preparation; no spectrum is passed here.

    Every enabled line requires a positive finite rest wavelength. Disabled
    entries may use null, as for the pending OII doublet reference. This
    function validates catalogue data, not support for a particular fitting
    model. The fitting stage must explicitly support the requested feature.
    It does not select lines, convert units or air/vacuum wavelengths, check
    observed coverage, or set continuum and fitting windows.
    """
    path = Path(path).expanduser()
    try:
        with path.open("r", encoding="utf-8") as stream:
            catalog = yaml.load(stream, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(catalog, dict):
        raise ValueError("The catalogue must be a YAML mapping.")
    version = catalog.get("schema_version")
    if type(version) is not int or version != 1:
        raise ValueError("schema_version must be the integer 1.")
    _require_text(catalog.get("catalog_id"), "catalog_id")
    unit_name = catalog.get("wavelength_unit")
    _require_text(unit_name, "wavelength_unit")
    try:
        unit = u.Unit(unit_name)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Unrecognized wavelength_unit: {unit_name!r}.") from exc
    if not unit.is_equivalent(u.m):
        raise ValueError("wavelength_unit must be a unit of length.")
    convention = catalog.get("wavelength_convention")
    if convention not in ("air", "vacuum"):
        raise ValueError("wavelength_convention must be 'air' or 'vacuum'.")

    sources = catalog.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("sources must be a nonempty mapping.")
    for source_id, source in sources.items():
        _require_text(source_id, "source identifier")
        if not isinstance(source, dict):
            raise ValueError(f"sources.{source_id} must be a mapping.")
        _require_text(source.get("description"), f"sources.{source_id}.description")
        if not any(isinstance(source.get(key), str) and source[key].strip()
                   for key in ("document", "url")):
            raise ValueError(f"sources.{source_id} requires a document or URL reference.")

    lines = catalog.get("lines")
    if not isinstance(lines, dict) or not lines:
        raise ValueError("lines must be a nonempty mapping.")
    for line_id, line in lines.items():
        _require_text(line_id, "line identifier")
        field = f"lines.{line_id}"
        if not isinstance(line, dict):
            raise ValueError(f"{field} must be a mapping.")
        _require_text(line.get("label"), f"{field}.label")
        if type(line.get("enabled")) is not bool:
            raise ValueError(f"{field}.enabled must be a YAML boolean.")
        if line.get("preferred_arm") not in ("blue", "red"):
            raise ValueError(f"{field}.preferred_arm must be 'blue' or 'red'.")
        if line.get("feature_type") not in ("individual_line", "doublet"):
            raise ValueError(f"{field}.feature_type must be 'individual_line' or 'doublet'.")
        source_id = line.get("wavelength_source")
        _require_text(source_id, f"{field}.wavelength_source")
        if source_id not in sources:
            raise ValueError(f"{field}.wavelength_source refers to an undefined source.")
        if "rest_wavelength" not in line:
            raise ValueError(f"{field}.rest_wavelength is required; use null if pending.")
        wavelength = line["rest_wavelength"]
        if line["enabled"] or wavelength is not None:
            _require_wavelength(wavelength, f"{field}.rest_wavelength")
        else:
            _require_text(line.get("status"), f"{field}.status")
        if line["feature_type"] == "doublet":
            components = line.get("components")
            if not isinstance(components, list) or len(components) != 2:
                raise ValueError(f"{field}.components must contain two wavelengths.")
            for index, value in enumerate(components):
                _require_wavelength(value, f"{field}.components[{index}]")
            if components[0] >= components[1]:
                raise ValueError(f"{field}.components must be distinct and increasing.")
    return catalog
