import re

def merge_lines(text):
    return re.sub(r'(?<!\n)\n(?!\n)', ' ', text)