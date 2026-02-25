class ConversationMemory:
    def __init__(self, max_turns=50):
        self.history = []
        self.max_turns = max_turns
        self.user_turn_count = 0
        self.diagnosis_complete = False
        self.waiting_remedies_consent = False
        self.last_diagnosis = None

    def mark_complete(self):
        self.diagnosis_complete = True

    def add_turn(self, role, content):
        if role == "user":
            self.user_turn_count += 1
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def get_formatted_history(self):
        formatted = ""
        for turn in self.history:
            formatted += f"{turn['role'].upper()}: {turn['content']}\n"
        return formatted.strip()

    def clear(self):
        self.history = []