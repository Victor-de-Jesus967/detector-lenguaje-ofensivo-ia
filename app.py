"""Aplicación web: predicción orientativa de lenguaje ofensivo en inglés."""
from pathlib import Path
import json
import time

import joblib
import streamlit as st
from src.pipeline import limpiar_texto

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Detector de lenguaje ofensivo',page_icon='🛡️',layout='centered')
st.title('🛡️ Detector de lenguaje ofensivo')
st.caption('Prototipo académico | Inglés (TweetEval) y español regional (piloto sintético)')
st.info('El modelo clasifica texto como ofensivo o no ofensivo. Su resultado es orientativo; no debe emplearse para sanciones o moderación automática sin revisión humana.')

@st.cache_resource
def get_model(path: str):
    return joblib.load(path)

idioma = st.selectbox('Selecciona el modelo', [
    'Español mexicano (piloto Veracruz/Tabasco)', 'Inglés (TweetEval — Offensive)'
])
regional = idioma.startswith('Español')
model_path = ROOT / 'models' / ('detector_regional.joblib' if regional else 'detector.joblib')
if regional:
    st.warning('Módulo experimental: entrenado con pocos ejemplos sintéticos. No validado con hablantes ni datos reales de Veracruz o Tabasco.')
if not model_path.exists():
    command = 'python -m src.train_regional' if regional else 'python -m src.download_data && python -m src.train'
    st.info(f'Modelo aún no encontrado. Ejecuta: `{command}`')
else:
    modelo = get_model(str(model_path))
    text = st.text_area('Escribe un mensaje para analizar:',height=150,
                        placeholder='Ej.: Qué onda, cabrón, ¿cómo estás?' if regional else 'Write a message to classify...')
    if st.button('Analizar mensaje',type='primary'):
        if not text.strip():
            st.warning('Introduce un mensaje antes de analizar.')
        else:
            start = time.perf_counter()
            proba = modelo.predict_proba([limpiar_texto(text)])[0]
            pred = int(modelo.classes_[proba.argmax()])
            elapsed_ms = (time.perf_counter()-start)*1000
            if pred == 1:
                st.error('Clasificación estimada: OFENSIVO')
            else:
                st.success('Clasificación estimada: NO OFENSIVO')
            idx = list(modelo.classes_).index(1)
            st.metric('Probabilidad estimada de clase ofensiva',f'{float(proba[idx])*100:.1f}%')
            st.caption(f'Latencia de esta solicitud: {elapsed_ms:.2f} ms')
            st.caption('Probabilidad no calibrada; el sistema puede equivocarse. No usar para sancionar personas.')

metrics_path = ROOT / 'results' / 'metricas.json'
if metrics_path.exists() and not regional:
    with st.expander('Resultados experimentales del conjunto de prueba'):
        metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        a,b = st.columns(2)
        a.metric('F1 macro',f"{metrics['test']['f1_macro']:.3f}")
        b.metric('Exactitud',f"{metrics['test']['accuracy']:.3f}")
        img = ROOT / 'results' / 'matriz_confusion.png'
        if img.exists():
            st.image(str(img),caption='Matriz de confusión en prueba')

regional_metrics = ROOT / 'results' / 'metricas_regionales.json'
if regional and regional_metrics.exists():
    with st.expander('Evaluación piloto de español (datos sintéticos)'):
        data = json.loads(regional_metrics.read_text(encoding='utf-8'))
        st.warning(data['aviso'])
        st.metric('F1 macro del conjunto sintético de prueba', f"{data['f1_test_macro']:.3f}")
        st.write('Matriz de confusión (filas reales, columnas predichas):')
        st.table(data['matriz_confusion'])
