import re

def remove_front_matter(text):
    """
    Removes front matter (TOC, Index, Foreword, etc.) by looking for the 
    actual start of the first chapter/section content.
    For this specific book, it looks for the first Srotas Roga section 
    that appears after the Index.
    """
    # Specific marker for this book where Chapter 1 starts
    marker = "Pranavaha Srotas Roga"
    
    # We find the LAST occurrence of the marker if it appears in the Index too.
    # Or we can search for a combination that likely identifies the content start.
    content_markers = [
        "Pranavaha Srotas Roga",
        "Brief Introduction of the disease: Kasa",
        "Kasa (Cough)"
    ]
    
    start_pos = -1
    for m in content_markers:
        # We look for the marker. To avoid the Index, we can look for it 
        # occurring after a certain amount of text or containing specific formatting.
        # But a safer way here is to find the marker that is NOT followed by a page number immediately.
        matches = list(re.finditer(re.escape(m), text))
        if matches:
            # Usually the last one is the actual heading if it appears in TOC multiple times
            # In our case, it appeared in Index early on.
            # Let s take the one that is farthest in the text but before the 50% mark 
            # (since it is Chapter 1) or just the one after the string "INTRODUCTION".
            
            # Better heuristic: Find the marker that appears after "INTRODUCTION" (the general one)
            intro_pos = text.find("\nINTRODUCTION\n")
            if intro_pos != -1:
                # Search for marker after general intro
                match_after_intro = re.search(re.escape(m), text[intro_pos:])
                if match_after_intro:
                    start_pos = intro_pos + match_after_intro.start()
                    break
            else:
                # Fallback to last match
                start_pos = matches[-1].start()
                break

    if start_pos != -1:
        # We might want to keep the "INTRODUCTION" if it s the start of content.
        # But the user specifically asked "before chapter 1".
        # Kasa is Chapter 1.
        return text[start_pos:]
    
    return text

