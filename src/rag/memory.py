import re


class ConversationMemory:
    def __init__(self, max_turns=50):
        self.history = []
        self.max_turns = max_turns
        self.user_turn_count = 0
        self.diagnosis_complete = False
        self.remedies_provided = False
        self.waiting_remedies_consent = False
        self.waiting_more_info_consent = False  # Track if user was asked about more info
        self.waiting_treatment_risk_profile = False
        self.treatment_risk_profile_collected = False
        self.last_diagnosis = None
        self.patient_age = None
        self.patient_gender = None

    def mark_complete(self):
        self.diagnosis_complete = True

    def add_turn(self, role, content):
        if role == "user":
            self.user_turn_count += 1
            self._update_patient_profile(content)
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def _update_patient_profile(self, content):
        text = (content or "").lower()

        # Gender capture.
        if any(g in text for g in ["female", "woman", "girl"]):
            self.patient_gender = "female"
        elif any(g in text for g in ["male", "man", "boy"]):
            self.patient_gender = "male"

        # Age capture.
        match = re.search(r"\b(\d{1,3})\b", text)
        if match:
            try:
                age = int(match.group(1))
                if 0 < age < 120:
                    self.patient_age = age
            except Exception:
                pass

    def get_formatted_history(self):
        formatted = ""
        if self.patient_age is not None or self.patient_gender is not None:
            age_text = str(self.patient_age) if self.patient_age is not None else "unknown"
            gender_text = self.patient_gender if self.patient_gender is not None else "unknown"
            formatted += f"PATIENT_PROFILE: age={age_text}, gender={gender_text}\n"

        for turn in self.history:
            formatted += f"{turn['role'].upper()}: {turn['content']}\n"
        return formatted.strip()

    def clear(self):
        self.history = []
