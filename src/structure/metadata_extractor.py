import re

DOSHA_KEYWORDS = {
    "Vata": ["vata"],
    "Pitta": ["pitta"],
    "Kapha": ["kapha"],
}

CATEGORY_KEYWORDS = {
    "Disease": ["roga", "disease", "disorder"],
    "Herb": ["herb", "plant", "dravya"],
    "Theory": ["principle", "theory", "concept"],
}

DISEASE_TYPE_KEYWORDS = {
    "Ano-rectal": ["ano-rectal", "fistula", "hemorrhoids", "fissure", "guda"],
    "Psychiatric": ["psychiatric", "insomnia", "epilepsy", "manas", "anidra", "apasmara"],
}

SROTAS_KEYWORDS = {
    "Purishavaha": ["purishavaha", "stool", "feces", "rectum", "intestine"],
    "Manovaha": ["manovaha", "mind", "memory", "consciousness", "sleep"],
}

TREATMENT_TYPE_KEYWORDS = {
    "Shodhana": ["shodhana", "vamana", "virechana", "basti", "nasya"],
    "Shamana": ["shamana", "linctus", "powder", "tablet"],
}

LEVEL_OF_CARE_KEYWORDS = {
    "PHC": ["phc", "solo ayurveda physician"],
    "CHC": ["chc", "small hospitals"],
}

FORMULATION_TYPE_KEYWORDS = {
    "Churna": ["churna", "powder"],
    "Vati": ["vati", "tablet", "gutika"],
    "Ghrita": ["ghrita", "ghee"],
}


def detect_dosha(text: str):
    text_lower = text.lower()
    for dosha, keywords in DOSHA_KEYWORDS.items():
        if any(k in text_lower for k in keywords):
            return dosha
    return None


def detect_category(text: str):
    text_lower = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(k in text_lower for k in keywords):
            return category
    return None


def detect_disease_type(text: str):
    text_lower = text.lower()
    for dtype, keywords in DISEASE_TYPE_KEYWORDS.items():
        if any(k in text_lower for k in keywords):
            return dtype
    return None


def detect_srotas(text: str):
    text_lower = text.lower()
    for srotas, keywords in SROTAS_KEYWORDS.items():
        if any(k in text_lower for k in keywords):
            return srotas
    return None


def detect_treatment_type(text: str):
    text_lower = text.lower()
    for ttype, keywords in TREATMENT_TYPE_KEYWORDS.items():
        if any(k in text_lower for k in keywords):
            return ttype
    return None


def detect_level_of_care(text: str):
    text_lower = text.lower()
    for level, keywords in LEVEL_OF_CARE_KEYWORDS.items():
        if any(k in text_lower for k in keywords):
            return level
    return None


def detect_formulation_type(text: str):
    text_lower = text.lower()
    for ftype, keywords in FORMULATION_TYPE_KEYWORDS.items():
        if any(k in text_lower for k in keywords):
            return ftype
    return None


def extract_topic(text: str):
    # Basic heuristic: first sentence
    first_sentence = re.split(r'[.!?]', text)[0]
    return first_sentence[:120]  # limit length