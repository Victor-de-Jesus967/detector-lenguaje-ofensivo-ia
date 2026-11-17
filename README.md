# Detector de lenguaje ofensivo — Actividades 4.1 y 4.2

Proyecto académico reproducible. **Idioma del dataset: inglés.** Modelo: TF-IDF + regresión logística. Interfaz: Streamlit.

## Requisitos
Python 3.10 a 3.13, conexión a Internet solo para descargar datos, y terminal Windows, Linux o macOS.

## Instalación y ejecución (Windows PowerShell)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.download_data
python -m src.train
python -m streamlit run app.py
```

En Linux/macOS, usar `python3 -m venv .venv`, `source .venv/bin/activate` y los mismos comandos `python -m ...`.

La interfaz suele estar disponible en http://localhost:8501.

## Organización
- `src/download_data.py`: descarga reproducible de TweetEval — offensive, repositorio oficial.
- `src/pipeline.py`: limpieza mínima conservadora.
- `src/train.py`: TF-IDF, búsqueda de hiperparámetros por validación cruzada en train, evaluación en validación y test, exportación de métricas y modelo.
- `app.py`: interfaz local Streamlit.
- `models/detector.joblib`: se produce después de entrenar, no se incluye en el ZIP.
- `results/metricas.json`, `results/matriz_confusion.png`: evidencias generadas tras ejecutar entrenamiento.
- `docs/guion_video.md`: estructura para Actividad 4.2.
- `tests`: pruebas de preprocesamiento.

## Criterios metodológicos
- Se respetan las particiones oficiales train / val / test.
- GridSearchCV selecciona hiperparámetros con 3 pliegues de train. Validation se reporta por separado. Test se evalúa una sola vez tras elegir modelo.
- F1 macro permite ponderar ambas clases sin ocultar desbalance de etiquetas.
- Se mide latencia CPU por texto en lotes y tiempo total de entrenamiento.
- TF-IDF se ajusta dentro del Pipeline, evitando fuga de información entre pliegues.
- La clasificación **no es detección universal de discurso de odio** ni inferencia de intención. Se requieren evaluaciones adicionales de sesgo, dialecto, dominio y español.
- No cargar artefactos joblib de origen desconocido: son archivos serializados capaces de ejecutar código al cargarse.

## Datos y atribución
TweetEval, tarea `offensive`:
https://github.com/cardiffnlp/tweeteval/tree/main/datasets/offensive

Barbieri, F., Camacho-Collados, J., Espinosa-Anke, L., & Neves, L. (2020). TweetEval: Unified benchmark and comparative evaluation for tweet classification. *Findings of the Association for Computational Linguistics: EMNLP 2020*, 1644–1650. https://doi.org/10.18653/v1/2020.findings-emnlp.148

## Pruebas rápidas

```bash
python -m unittest discover -s tests -v
```

## Alcance
Prototipo demostrativo académico, no desplegar públicamente sin autenticación, límites de uso y política de tratamiento de datos. No registrar mensajes personales en logs sin autorización. Las métricas no vienen prellenadas: deben obtenerse ejecutando el entrenamiento.
