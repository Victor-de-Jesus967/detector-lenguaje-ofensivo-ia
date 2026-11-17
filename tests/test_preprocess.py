import unittest
from src.pipeline import limpiar_texto

class TextTests(unittest.TestCase):
    def test_normaliza_espacios(self):
        self.assertEqual(limpiar_texto(' hello  world\n'), 'hello world')
    def test_url(self):
        self.assertEqual(limpiar_texto('visit https://example.com/path'), 'visit URL')
    def test_preserva_negacion(self):
        self.assertIn('not',limpiar_texto('not offensive'))

if __name__ == '__main__':
    unittest.main()
