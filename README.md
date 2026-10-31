# Mapeo de inundaciones con Sentinel-1 y U-Net (Sen1Floods11 + Lima 2017)

Repositorio de la Pregunta 2 del Examen Parcial de Redes Neuronales y Aprendizaje Profundo (UNI-FIIS, 2026-II).

## Un solo comando

```bash
pip install -r requirements.txt
bash run_all.sh            # descarga ~9 GB, Otsu, ablaciones x3 semillas, evaluación, fallos y tablas
QUICK=1 bash run_all.sh    # prueba de humo (2 epochs, pocos chips): comprueba que todo corre
PERU=1 bash run_all.sh     # además construye la región de Lima (si no existe) y evalúa en ella, sin reentrenar
```

Entorno probado: Google Colab con GPU T4 (Python 3.10+, PyTorch 2.x). La U-Net y todo el pipeline usan solo
PyTorch, NumPy, SciPy y tifffile; `rasterio`, `geopandas`, `pystac-client` y `planetary-computer` solo hacen falta
para construir la región de Lima.

## Estructura

| Archivo | Qué hace | Entrega del enunciado |
|---|---|---|
| `src/unet.py` | U-Net desde cero (concatenación, sin saltos o con compuertas de atención; profundidad 3/4) | 1, 2b |
| `src/data.py` | Lectura GeoTIFF, NaN → no válido, recorte en dB, normalización con estadísticas del train, Lee opcional, fuentes dorada/débil | 1, 2a |
| `src/losses.py` | BCE con `pos_weight`, Dice, combinada y focal, con píxeles ignorados y peso por chip | 2b |
| `src/train.py` | Entrenamiento con semilla fija, selección por IoU en VALID, preentrenar débil → ajustar dorado | 2 |
| `src/evaluate.py` | IoU, F1, precisión, recall, omisión/comisión, matriz de confusión, por evento y por chip; línea base Otsu | 3, 4 |
| `src/gee_export_lima.py` | Exporta Sentinel-1 de Lima (17-mar-2017) desde Earth Engine en el formato de Sen1Floods11 | 4 |
| `src/peru_prep.py` | Corta la escena en chips 512×512 y rasteriza los polígonos EMSN038 | 4 |
| `src/failure_analysis.py` | Etiquetas heurísticas de error (doble rebote, superficie oscura/sombra, speckle, borde) + figuras | 5 |
| `src/make_tables.py` | Tablas media ± desv. entre semillas y brecha de generalización | 2–4 |
| `src/lima_etiquetas.py`, `src/lima_s1.py`, `src/lima_otsu.py`, `src/lima_mapa.py` | Región de Lima: EMSN038 + Sentinel-1B del 17-mar-2017, línea base Otsu y mapa | 4 |
| `notebook/Pregunta2_UNet_Lima_Colab.ipynb` | Cuaderno único para Colab (GPU T4): clona este repo y corre todo | 1–6 |
| `resultados/` | Figuras de nuestros resultados y el script que las genera | 2–5 |

## Resultados del grupo (IoU de la clase agua)

| Modelo | Test (eventos vistos) | Bolivia (región nueva) | Lima (región construida) |
|---|---|---|---|
| U-Net completa (`A_dorado`, Colab T4, semillas 0 y 1) | 0.658 / 0.656 | 0.645 / 0.670 | — |
| U-Net reducida con saltos (CPU, 1 semilla) | 0.647 | 0.578 | — |
| U-Net reducida sin saltos (CPU, 1 semilla) | 0.628 | 0.624 | — |
| Otsu-VH (sin entrenar) | 0.554 | 0.718 | 0.010 |

Lima: 51 chips de 512×512, 1.25 % de píxeles con agua; el umbral de Otsu de la escena es −16.9 dB (en Sen1Floods11, de −20
a −25 dB). Detalle y límites en el informe.

## Datos

`python -m src.download --dest data/sen1floods11 --with-weak --with-s2` baja desde el bucket público
`gs://sen1floods11/v1.1` (sin cuenta) solo lo necesario: S1Hand, LabelHand, S1OtsuLabelHand, JRCWaterHand, S2Hand,
S1Weak, S1OtsuLabelWeak y S2IndexLabelWeak. S2Weak (25.7 GB) no se usa.

Particiones oficiales (CSV del bucket): train 252, valid 89, test 90 y Bolivia 15 chips dorados. Las etiquetas débiles
de **Bolivia se excluyen** del entrenamiento (`--exclude-weak-events Bolivia`), para que Bolivia sea de verdad una
región no vista.

## Región de prueba de Lima (automática con `PERU=1`, sin cuentas)

1. `python -m src.lima_etiquetas --dir data/lima`: descarga la geodatabase oficial de EMSN038 (~2.3 GB) y extrae la
   capa `Potential_Maximum_flood` (inundación) y la zona rotulada (AOI ∩ huellas WorldView-2 del 18 y 21-mar-2017).
2. `python -m src.lima_s1 --out data/lima/lima_s1_20170317.tif`: escena Sentinel-1B del 17-mar-2017 (VV, VH) desde
   Microsoft Planetary Computer, reproyectada a EPSG:4326 con el píxel de Sen1Floods11 y pasada de γ0 a σ0 en dB con el
   ángulo de incidencia de la anotación del producto.
3. `python -m src.peru_prep ... --root data/sen1floods11/v1.1`: chips de 512×512 y etiquetas rasterizadas (51 chips).
4. `python -m src.lima_otsu --root data/sen1floods11/v1.1`: línea base Otsu-VH de la escena.
5. Evaluación de todos los modelos ya entrenados y `src.lima_mapa` (figura). Ruta alternativa por Earth Engine:
   `src/gee_export_lima.py`.

## Reproducibilidad

- Semilla global y generador propio del `DataLoader`: dos corridas con la misma semilla usan los mismos recortes.
- Cada configuración se corre con 3 semillas y se reporta media ± desviación estándar.
- El checkpoint se elige por IoU en VALID. El umbral también se elige en VALID (`--tune-threshold`). Test, Bolivia y Lima
  no se usan para tomar ninguna decisión.
