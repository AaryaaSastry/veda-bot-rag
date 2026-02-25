from dataclasses import dataclass
from typing import Optional

@dataclass
class KnowledgeChunk:
    text: str
    source: str
    chapter: Optional[str] = None
    topic: Optional[str] = None
    dosha: Optional[str] = None
    category: Optional[str] = None
    disease_type: Optional[str] = None
    srotas: Optional[str] = None
    treatment_type: Optional[str] = None
    level_of_care: Optional[str] = None
    formulation_type: Optional[str] = None

    def to_dict(self):
        return {
            "text": self.text,
            "source": self.source,
            "chapter": self.chapter,
            "topic": self.topic,
            "dosha": self.dosha,
            "category": self.category,
            "disease_type": self.disease_type,
            "srotas": self.srotas,
            "treatment_type": self.treatment_type,
            "level_of_care": self.level_of_care,
            "formulation_type": self.formulation_type,
        }