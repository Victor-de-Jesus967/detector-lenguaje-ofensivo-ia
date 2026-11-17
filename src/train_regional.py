"""Experimento piloto EN ESPAÑOL, separado del benchmark inglés.

Datos pequeños y sintéticos: resultados ilustrativos, NO generalizables.
"""
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]


def main():
    df = pd.read_csv(ROOT/'data'/'regional'/'ejemplos_piloto.csv', encoding='utf-8')
    train = df[df['particion']=='train']
    test = df[df['particion']=='test']
    if set(train['texto']) & set(test['texto']):
        raise ValueError('Hay textos repetidos entre entrenamiento y prueba')
    pipeline = Pipeline([
       ('tfidf', TfidfVectorizer(strip_accents='unicode', lowercase=True,
                                analyzer='char', ngram_range=(2,5), sublinear_tf=True)),
       ('classifier', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42))
    ])
    grid = GridSearchCV(pipeline, {'classifier__C':[0.1,1,10]}, cv=StratifiedKFold(n_splits=4,shuffle=True,random_state=42), scoring='f1_macro', n_jobs=-1)
    start=time.perf_counter()
    grid.fit(train['texto'],train['etiqueta'])
    duration=time.perf_counter()-start
    predictions=grid.best_estimator_.predict(test['texto'])
    report={
      'aviso':'Experimento piloto con ejemplos sintéticos redactados; no mide desempeño regional real',
      'dataset':'Ejemplos originales de uso mexicano y expresiones contextualizadas de Veracruz/Tabasco',
      'cantidad_train':len(train),'cantidad_test':len(test),
      'hiperparametros':grid.best_params_,'f1_cv_macro':float(grid.best_score_),
      'f1_test_macro':float(f1_score(test['etiqueta'],predictions,average='macro',zero_division=0)),
      'accuracy_test':float(accuracy_score(test['etiqueta'],predictions)),
      'matriz_confusion':confusion_matrix(test['etiqueta'],predictions,labels=[0,1]).tolist(),
      'detalle':classification_report(test['etiqueta'],predictions,output_dict=True,zero_division=0),
      'segundos_entrenamiento':round(duration,3)
    }
    (ROOT/'models').mkdir(exist_ok=True)
    (ROOT/'results').mkdir(exist_ok=True)
    joblib.dump(grid.best_estimator_,ROOT/'models'/'detector_regional.joblib')
    (ROOT/'results'/'metricas_regionales.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
