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
st.caption('Prototipo académico | Inglés | TweetEval — Offensive')
st.info('El modelo clasifica texto como ofensivo o no ofensivo. Su resultado es orientativo; no debe emplearse para sanciones o moderación automática sin revisión humana.')

@st.cache_resource
def get_model():
    return joblib.load(ROOT / 'models' / 'detector.joblib')

if not (ROOT / 'models' / 'detector.joblib').exists():
    st.warning('Modelo no encontrado. Ejecuta `python -m src.download_data` y después `python -m src.train`.')
    st.stop()

modelo = get_model()
text = st.text_area('Escribe un mensaje en inglés para analizar:',height=150,placeholder='Write a message to classify...')
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
        st.caption('La probabilidad es una salida del modelo; no equivale a certeza ni garantiza calibración.')

metrics_path = ROOT / 'results' / 'metricas.json'
if metrics_path.exists():
    with st.expander('Resultados experimentales del conjunto de prueba'):
        metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        a,b = st.columns(2)
        a.metric('F1 macro',f"{metrics['test']['f1_macro']:.3f}")
        b.metric('Exactitud',f"{metrics['test']['accuracy']:.3f}")
        img = ROOT / 'results' / 'matriz_confusion.png'
        if img.exists():
            st.image(str(img),caption='Matriz de confusión en prueba')
