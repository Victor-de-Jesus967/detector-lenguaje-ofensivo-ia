"""Entrenamiento verificable, selección sin fuga de datos y evaluación final."""
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.pipeline import limpiar_texto

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'raw'
RESULTS = ROOT / 'results'
MODELS = ROOT / 'models'


def load_split(prefix):
    # El repositorio oficial conserva una observación por línea.
    texts = (DATA / f'{prefix}_text.txt').read_text(encoding='utf-8').splitlines()
    labels = (DATA / f'{prefix}_labels.txt').read_text(encoding='utf-8').splitlines()
    if len(texts) != len(labels):
        raise ValueError(f'Diferente número de textos y etiquetas en {prefix}')
    return [limpiar_texto(x) for x in texts], np.array([int(y.strip()) for y in labels])


def main():
    for directory in (RESULTS, MODELS):
        directory.mkdir(parents=True, exist_ok=True)
    needed = [DATA / f'{part}_{kind}.txt' for part in ('train', 'val', 'test') for kind in ('text','labels')]
    if any(not p.exists() for p in needed):
        raise SystemExit('Faltan datos. Ejecuta primero: python -m src.download_data')

    X_train, y_train = load_split('train')
    X_val, y_val = load_split('val')
    X_test, y_test = load_split('test')
    print('Tamaños:', {'train':len(y_train), 'validation':len(y_val), 'test':len(y_test)})
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(lowercase=True, ngram_range=(1,2), sublinear_tf=True, max_features=50000)),
        ('classifier', LogisticRegression(max_iter=1500, random_state=42, class_weight='balanced')),
    ])
    grid = GridSearchCV(
        pipeline,
        param_grid={'tfidf__min_df':[2,4], 'classifier__C':[0.5,1.0,2.0]},
        scoring='f1_macro',
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        n_jobs=-1, refit=True, verbose=1,
    )
    t0 = time.perf_counter()
    grid.fit(X_train, y_train)
    training_seconds = time.perf_counter() - t0
    model = grid.best_estimator_
    y_val_pred = model.predict(X_val)
    # Se evalúa test una sola vez, después de seleccionar hiperparámetros en train.
    y_test_pred = model.predict(X_test)
    report = {
        'dataset': 'cardiffnlp/tweeteval — offensive (idioma inglés)',
        'split_sizes': {'train':len(y_train),'validation':len(y_val),'test':len(y_test)},
        'best_params':grid.best_params_,
        'cross_validation_macro_f1':float(grid.best_score_),
        'validation_macro_f1':float(f1_score(y_val,y_val_pred,average='macro',zero_division=0)),
        'test': {
            'accuracy':float(accuracy_score(y_test,y_test_pred)),
            'precision_macro':float(precision_score(y_test,y_test_pred,average='macro',zero_division=0)),
            'recall_macro':float(recall_score(y_test,y_test_pred,average='macro',zero_division=0)),
            'f1_macro':float(f1_score(y_test,y_test_pred,average='macro',zero_division=0)),
            'f1_offensive':float(f1_score(y_test,y_test_pred,pos_label=1,zero_division=0)),
            'report':classification_report(y_test,y_test_pred,target_names=['No ofensivo','Ofensivo'],output_dict=True,zero_division=0),
            'confusion_matrix':confusion_matrix(y_test,y_test_pred,labels=[0,1]).tolist(),
        },
        'training_seconds':round(training_seconds,3),
    }
    # Medición de inferencia en CPU; la primera ejecución no se incluye (warm-up).
    batch = X_test[:min(100,len(X_test))]
    _ = model.predict(batch)
    times = []
    for _ in range(20):
        start = time.perf_counter()
        _ = model.predict(batch)
        times.append(1000*(time.perf_counter()-start)/len(batch))
    report['latency_ms_per_text_cpu'] = {'median':float(np.median(times)), 'p95':float(np.percentile(times,95)), 'batch_size':len(batch)}
    joblib.dump(model, MODELS / 'detector.joblib')
    (RESULTS / 'metricas.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    cm = confusion_matrix(y_test,y_test_pred,labels=[0,1])
    ConfusionMatrixDisplay(cm,display_labels=['No ofensivo','Ofensivo']).plot(cmap='Blues',values_format='d')
    plt.title('Matriz de confusión — conjunto de prueba')
    plt.tight_layout()
    plt.savefig(RESULTS / 'matriz_confusion.png',dpi=170)
    plt.close()
    print(json.dumps({k:v for k,v in report.items() if k != 'test'},ensure_ascii=False,indent=2))
    print('F1 macro de prueba:',report['test']['f1_macro'])
    print('Modelo y resultados guardados.')


if __name__ == '__main__':
    main()
