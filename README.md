# annular-redshift

Paquete Python en desarrollo para medir redshifts de líneas de emisión en
espectros anulares SAMI y sus diferencias entre regiones.

## Estado del desarrollo


La configuración acordada utiliza `BIN_MASK=1` como región central y
`BIN_MASK=4` como región externa.

## Requisitos

- Python 3.10 o superior, con `pip` y `venv` disponibles.
- La carpeta del proyecto, con `pyproject.toml` en su raíz.

## Instalación en un entorno virtual (Linux y macOS)

Un entorno virtual mantiene las dependencias de este proyecto separadas de
las de otros proyectos. Ejecutar los siguientes comandos en una terminal.

### 1. Entrar en el proyecto

Desde la carpeta que contiene el proyecto:

```bash
cd annular-redshift
```

En ella debe estar
`pyproject.toml`.

### 2. Crear el entorno

```bash
conda ....
```

Comprobar que la versión es 3.10 o superior.

### 3. Activarlo

```bash
conda activate ...
```


### 4. Instalar el paquete en modo editable

```bash
python -m pip install -e .
```

El punto indica la carpeta actual. `-e` permite desarrollar el código sin
reinstalarlo después de cada modificación de los archivos Python. Reiniciá
la sesión de Python o el kernel del notebook para cargar los cambios.
Si cambiás dependencias en `pyproject.toml`, repetí la instalación.

Para instalar también las herramientas opcionales de pruebas y documentación:

```bash
python -m pip install -e ".[test,docs]"
```


## Estructuras de datos

| Clase | Contenido |
| --- | --- |
| `AnnularCube` | Flujo y varianza tridimensionales, eje espectral, etiquetas de anillos, unidades y metadatos de un brazo espectral. |
| `Spectrum` | Flujo y varianza unidimensionales de una región, longitudes de onda, máscara opcional y metadatos. |

