import csv
from pathlib import Path


def test_data_labels_and_splits():
    path = Path(__file__).resolve().parents[1] / 'data' / 'regional' / 'ejemplos_piloto.csv'
    with path.open(encoding='utf-8',newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 80
    assert {r['etiqueta'] for r in rows} == {'0','1'}
    assert {r['particion'] for r in rows} == {'train','test'}
    assert not ({r['texto'] for r in rows if r['particion']=='train'} & {r['texto'] for r in rows if r['particion']=='test'})
