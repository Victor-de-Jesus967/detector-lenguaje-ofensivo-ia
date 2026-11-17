"""Descarga la partición 'offensive' del repositorio oficial de TweetEval."""
from pathlib import Path
from urllib.request import urlretrieve

BASE = 'https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/offensive'
DEST = Path(__file__).resolve().parents[1] / 'data' / 'raw'
FILES = ('train_text.txt', 'train_labels.txt', 'val_text.txt', 'val_labels.txt', 'test_text.txt', 'test_labels.txt')


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = DEST / name
        if not path.is_file():
            print('Descargando', name)
            urlretrieve(f'{BASE}/{name}', path)
        else:
            print('Ya existe:', name)
    print('Datos disponibles en', DEST)


if __name__ == '__main__':
    main()
