import re

def remove_page_numbers(pages):
    """
    Removes standalone page numbers (Roman or Arabic) from top/bottom lines.
    """
    # Fixed regex: Inline flag (?i) replaced with re.IGNORECASE or placed correctly
    page_num_pattern = re.compile(r"^(?:[ivxlc]+|\d+)$", re.IGNORECASE)

    cleaned_pages = []
    for page in pages:
        lines = page.split("\n")
        if not lines:
            cleaned_pages.append(page)
            continue
            
        new_lines = []
        total_lines = len(lines)
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Check if it is a standalone page number at the extremities (first 3 or last 3 lines)
            if (i < 3 or i > total_lines - 4) and page_num_pattern.match(stripped):
                continue
            
            new_lines.append(line)
            
        cleaned_pages.append("\n".join(new_lines))
        
    return cleaned_pages

