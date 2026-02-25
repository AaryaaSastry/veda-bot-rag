import re

def fix_hyphenation(text):
    return re.sub(r'-\n(\w+)', r'\1', text)