"""Contenedores de datos para cubos anulares y espectros individuales.

Estas clases no leen FITS, extraen anillos ni ajustan líneas. La validación
del formato y la curación se implementarán en sus respectivos módulos.
Los arrays se conservan por referencia: estas clases no los copian ni escalan.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


@dataclass
class AnnularCube:
    """Datos de un producto anular de un brazo del espectrógrafo.

    Parameters
    ----------
    flux : numpy.ndarray
        Flujo con ejes (longitud de onda, y, x).
    variance : numpy.ndarray
        Varianza del flujo, con la misma forma que ``flux``. Su unidad debe
        ser el cuadrado de ``flux_unit``; no es el error estándar.
    wavelength : numpy.ndarray
        Longitudes de onda originales, con forma (n_wave,).
    bin_mask : numpy.ndarray
        Etiquetas originales de las regiones, con forma (ny, nx).
        No se renumeran: se usarán las regiones 1 y 4 del notebook.
    wavelength_unit : str
        Unidad del eje espectral, verificada por el loader.
    flux_unit : str or None
        Unidad del flujo. None indica que no fue determinada.
    z_initial : float or None
        Redshift de referencia para localizar líneas. No es el redshift
        medido de un anillo. None indica que no está disponible.
    galaxy_id : str or None
        Identificador de la galaxia.
    arm : {'blue', 'red'} or None
        Brazo espectral al que pertenece el producto.
    wavelength_convention : {'air', 'vacuum', 'unknown'}
        Convención de longitud de onda. No se presupone aire o vacío.
    spectral_frame : str
        Sistema de referencia espectral indicado por la procedencia.
        'unknown' significa que aún no fue determinado.
    metadata : dict
        Procedencia, encabezados y otra información conservada por el loader.
    """

    flux: NDArray[np.floating]
    variance: NDArray[np.floating]
    wavelength: NDArray[np.floating]
    bin_mask: NDArray[np.generic]
    wavelength_unit: str
    flux_unit: str | None = None
    z_initial: float | None = None
    galaxy_id: str | None = None
    arm: Literal["blue", "red"] | None = None
    wavelength_convention: Literal["air", "vacuum", "unknown"] = "unknown"
    spectral_frame: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Spectrum:
    """Espectro de una región, con sus unidades y varianza.

    ``wavelength``, ``flux`` y ``variance`` son arrays de forma (n_wave,).
    ``region_id`` conserva la etiqueta de BIN_MASK. ``mask`` es una máscara
    booleana de igual longitud: True significa canal excluido. None indica
    que aún no se asignó una máscara; no certifica que todos los canales
    sean válidos. La curación decidirá qué canales excluir.

    ``variance`` está expresada en el cuadrado de la unidad del flujo.
    Los metadatos pueden registrar el spaxel representativo y el número
    de spaxels de la región. Las demás convenciones coinciden con AnnularCube.
    """

    wavelength: NDArray[np.floating]
    flux: NDArray[np.floating]
    variance: NDArray[np.floating]
    region_id: int
    wavelength_unit: str
    flux_unit: str | None = None
    z_initial: float | None = None
    galaxy_id: str | None = None
    arm: Literal["blue", "red"] | None = None
    wavelength_convention: Literal["air", "vacuum", "unknown"] = "unknown"
    spectral_frame: str = "unknown"
    mask: NDArray[np.bool_] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def error(self) -> NDArray[np.float64]:
        """Error estándar sqrt(varianza); NaN para varianzas no válidas.

        No modifica la varianza original. Una varianza cero da error cero;
        el ajuste deberá excluir esos canales al construir sus pesos.
        Esta propiedad no aplica la máscara de canales.
        """
        variance = np.asarray(self.variance, dtype=float)
        error = np.full(variance.shape, np.nan, dtype=float)
        valid = np.isfinite(variance) & (variance >= 0)
        np.sqrt(variance, out=error, where=valid)
        return error
