import unicodedata
import re

def normalize(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()