"""Funciones compartidas entre entrenamiento e inferencia."""
import re
import html


def limpiar_texto(texto: str) -> str:
    """Limpieza conservadora: conserva negaciones y palabras potencialmente ofensivas."""
    texto = html.unescape(str(texto))
    texto = re.sub(r'https?://\S+|www\.\S+', ' URL ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto
