import re

def parse_chapters(text):
    """
    Parses the full text into chapters based on numerical titles (e.g., 1 Kasa, 2 Tamaka Swasa).
    Returns a list of dictionaries: [{"title": "Chapter Title", "content": "Chapter Text"}]
    """
    # Based on index: "1 Kasa", "2 Tamaka Swasa", "3 Amlapitta", etc.
    # Pattern looks for digits followed by a title at the start of a line
    # Note: TOC removal left us starting with "Pranavaha Srotas Roga"
    
    # We use a pattern that matches the chapter headings seen in the index but in the actual text body.
    # In the text, they appear like:
    # 1
    # Kasa
    # (sometimes)
    
    # Let s try to find patterns of DIGIT followed by TITLE
    # For this book, we know the list of chapters from the index we read earlier.
    
    # A more general approach for this specific book:
    chapter_markers = [
        "Kasa", "Tamaka Swasa", "Amlapitta", "Jalodara", "Amavata", "Jwara", "Pandu",
        "Ekakushtha", "Kamala", "Hypothyroidism", "Madhumeha", "Sthoulya", "Arsha", "Atisara",
        "Bhagandara", "Krimi", "Parikartika", "Anidra", "Apasmara", "Vishada", "Ashmari",
        "Mutraghata", "Mutrasthila", "Asrigdara", "Kashtaarthava", "Shwetapradara", "Avabahuka",
        "Katigraha", "Gridhrasi", "Pakshaghata", "Sandhigata Vata", "Vatarakta", "Abhishyanda",
        "Adhimantha", "Dantavestaka", "Mukhapaka", "Pratishyaya", "Shiroroga"
    ]
    
    # Create a regex to find these as standalone lines (or near start of lines)
    combined_pattern = "|".join([re.escape(m) for m in chapter_markers])
    # Heading usually appears on its own line after some whitespace or section title
    regex = re.compile(rf"^\s*(?:\d+\s*\n)?({combined_pattern})\s*$", re.IGNORECASE | re.MULTILINE)
    
    splits = list(regex.finditer(text))
    
    chapters = []
    for i, match in enumerate(splits):
        title = match.group(1).strip()
        start = match.end()
        end = splits[i+1].start() if i+1 < len(splits) else len(text)
        
        content = text[start:end].strip()
        if content:
            chapters.append({
                "title": title,
                "content": content
            })
    
    # Fallback: if no chapters found, treat the whole document as a single chapter
    if not chapters and text:
        chapters.append({
            "title": "General",
            "content": text
        })
            
    return chapters

