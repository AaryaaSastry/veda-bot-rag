import re
from collections import Counter

def remove_headers_footers(pages):
    """
    Intelligently identifies and removes recurring headers and footers.
    Based on frequency analysis of the first and last few lines across all pages.
    """
    if not pages:
        return pages

    # 1. Frequency analysis of top and bottom lines
    top_lines = []
    bottom_lines = []
    
    for page in pages:
        lines = [l.strip() for l in page.split("\n") if l.strip()]
        if lines:
            top_lines.append(lines[0])
            if len(lines) > 1:
                bottom_lines.append(lines[-1])

    # Find lines that appear in > 30% of pages (likely headers/footers)
    threshold = len(pages) * 0.3
    potential_headers = [line for line, count in Counter(top_lines).items() if count > threshold]
    potential_footers = [line for line, count in Counter(bottom_lines).items() if count > threshold]

    # Specific pattern for this book seen in inspection
    static_header = "AYURVEDIC STANDARD TREATMENT GUIDELINES"
    
    cleaned_pages = []
    for page in pages:
        lines = page.split("\n")
        new_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Remove if it matches frequent patterns or the specific observed header
            if stripped in potential_headers and (i < 3):
                continue
            if stripped in potential_footers and (i > len(lines) - 4):
                continue
            if static_header in stripped:
                # If the static header is merged on a line, we try to strip just the header part
                line = line.replace(static_header, "").strip()
                if not line: continue
            
            new_lines.append(line)
        
        cleaned_pages.append("\n".join(new_lines))
        
    return cleaned_pages

