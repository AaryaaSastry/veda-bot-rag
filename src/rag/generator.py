import config
import re
from google import genai


class Generator:

    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        #Using the gemma model for testing currently as it offers unlimited tokens
        #It is recomended to switch to a more powerful model such as Gemini-2.0-flash or Gemini-2.5-flash for final production use. 
        self.model_id = "gemma-3-27b-it"

    def generate_text(self, prompt):
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt
        )
        return response.text

    def _trim_history(self, conversation_history, max_lines=24):
        lines = [line for line in conversation_history.splitlines() if line.strip()]
        if len(lines) <= max_lines:
            return conversation_history
        return "\n".join(lines[-max_lines:])

    def _build_context(self, retrieved_chunks, max_chunks=6, max_chars_per_chunk=900):
        limited = retrieved_chunks[:max_chunks]
        parts = []
        for i, chunk in enumerate(limited):
            text = (chunk.get("text", "") or "").strip()
            if len(text) > max_chars_per_chunk:
                text = text[:max_chars_per_chunk].rstrip() + "..."
            parts.append(f"Source {i+1}:\n{text}")
        return "\n\n".join(parts)

    def _extract_previous_questions(self, conversation_history):
        questions = []
        for line in conversation_history.splitlines():
            if not line.startswith("ASSISTANT:"):
                continue
            content = line[len("ASSISTANT:"):].strip()
            if "?" in content:
                questions.append(content)
        return questions

    def _normalize_question(self, text):
        lowered = text.lower()
        lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
        lowered = re.sub(r"\s+", " ", lowered).strip()
        return lowered

    def _is_duplicate_question(self, candidate, previous_questions):
        cand_norm = self._normalize_question(candidate)
        cand_tokens = set(cand_norm.split())
        if not cand_norm or not cand_tokens:
            return False

        for prev in previous_questions:
            prev_norm = self._normalize_question(prev)
            if cand_norm == prev_norm:
                return True
            prev_tokens = set(prev_norm.split())
            if not prev_tokens:
                continue
            overlap = len(cand_tokens & prev_tokens) / max(1, len(cand_tokens | prev_tokens))
            if overlap >= 0.75:
                return True
        return False

    def generate_diagnosis(self, conversation_history, retrieved_chunks):
        trimmed_history = self._trim_history(conversation_history, max_lines=30)
        context = self._build_context(retrieved_chunks, max_chunks=8, max_chars_per_chunk=1100)
        prompt = f"""
SYSTEM:
You are an Ayurvedic clinical diagnostic expert.
Based ON THE RETRIEVED SOURCES and the conversation history, provide a potential diagnosis and the reasoning behind it.

CONVERSATION HISTORY:
{trimmed_history}

RETRIEVED SOURCES:
{context}

Format your output exactly like this:
DIAGNOSIS: [Name of condition]
REASONING: [Step by step reasoning based on symptoms and sources]
"""
        return self.generate_text(prompt)

    def verify_diagnosis(self, diagnosis_report, conversation_history):
        trimmed_history = self._trim_history(conversation_history, max_lines=30)
        prompt = f"""
SYSTEM:
You are a senior Ayurvedic medical reviewer.
Look at the following diagnosis, reasoning, and the full conversation history.
Ensure that you understand the age and gender of the patient from the conversation history and use that for further diagnosis.
CONVERSATION HISTORY:
{trimmed_history}

DIAGNOSIS REPORT:
{diagnosis_report}

Does this diagnosis make sense and is it sufficiently supported by the facts gathered from the user?
Check very strictly to ensure it checks the accuract of the diagnosis based on the conversation history and the symptoms mentioned, and ensure that the reasoning is sound and follows a logical flow.
Respond with "YES" if the diagnosis is valid and well supported, or "NO" if it is not.
Respond with ONLY one word: "YES" or "NO".
"""
        response = self.generate_text(prompt).strip().upper()
        return "YES" in response

    def generate(self, question, retrieved_chunks, conversation_history="", mode="gathering"):
        if not retrieved_chunks:
            yield "I could find no relevant information in the documents provided."
            return

        trimmed_history = self._trim_history(conversation_history, max_lines=24)
        if mode == "gathering":
            context = self._build_context(retrieved_chunks, max_chunks=4, max_chars_per_chunk=750)
        else:
            context = self._build_context(retrieved_chunks, max_chunks=6, max_chars_per_chunk=950)

        if mode == "gathering":
            instruction = """
STRICT OUTPUT RULES:
1. Output ONLY the question. Nothing else. No acknowledgments, no sympathy, no conversational filler.
2. DO NOT say things like "Thank you for providing your age and gender" or "It's interesting that..." or any similar statements.
3. DO NOT provide any context, explanations, or commentary before or after the question.
4. Start your response directly with the question word (What, How, When, Where, Which, etc.) or a direct question.
5. DO NOT REPEAT THE QUESTIONS ATALL , IF ONCE ASKED DO NOT ASK AGAIN, MOVE ON TO OTHER QUESTIONS.

QUESTION RULES:
1. Check conversation history. If age and gender are NOT mentioned yet, ask: "To provide an accurate Ayurvedic assessment, could you share your age and gender?"
2. If age and gender are already provided, ask ONLY ONE (1) clarifying question to help narrow down the diagnosis.
3. DO NOT provide treatments or final answers yet.
4. Ground your question strictly in the provided sources and conversation history.
5. Gather information about lifestyle, diet, medical history, and habits as this is crucial for Ayurvedic diagnosis.
6. Avoid asking questions that are not relevant to Ayurvedic diagnosis.

EXAMPLE OF WRONG OUTPUT: "Thank you for sharing that. Given your symptoms, I would like to ask..."
EXAMPLE OF CORRECT OUTPUT: "What time of day do your headaches typically occur?"
"""
        elif mode == "diagnosis":
            instruction = """
1. Provide a professional diagnosis summary from the report.
2. Use very minimal markdown (avoid bold and headers).
3. End your response by asking if the user would like to hear remedies, do's, and don'ts.
4. Give a clear understanding of the users current condition and places where they can improve based on the sources and the conversation history.
5. Avoid medical jargon as much as possible and make it user friendly and easy to understand.
6. Ensure u have an understanding of the user's lifestyle, diet, medical history, and habits as this is crucial for Ayurvedic diagnosis.
"""
        elif mode == "remedies":
            instruction = """
1. Extract and provide Ayurvedic remedies, treatments, and detailed 'Do's and Don'ts' directly from the RETRIEVED SOURCES.
2. If certain foods or habits are recommended or forbidden, list them clearly.
3. Formatting: NO bolding, NO complex headers. Use standard numbering and single line breaks.
4. Be comprehensive but organized without extra markdown fluff.
5. Strictly use given sources.
6. Provide the solution in a very clear consise manner that is user friendly and easy to understand, avoid medical jargon as much as possible.
"""
        else:
            instruction = "Provide a professional concluding response."

        prompt = f"""
SYSTEM:
You are an Ayurvedic clinical assistant.

GLOBAL RULES (APPLY TO ALL MODES):
- NO sympathy statements (e.g., "I understand this must be difficult")
- NO acknowledgments (e.g., "Thank you for sharing", "I appreciate that information")
- NO conversational filler (e.g., "It's interesting that...", "Given what you've told me...")
- NO citations or source references in your response
- Go directly to the content requested

CONVERSATION HISTORY:
{trimmed_history}

RETRIEVED SOURCES:
{context}

CURRENT USER INPUT:
{question}

INSTRUCTIONS:
{instruction}
- Groundedness: Remain strictly inside the provided sources.
- Ask questions only if they make sense for the patient's context.

Answer:
"""

        try:
            if mode == "gathering":
                previous_questions = self._extract_previous_questions(trimmed_history)
                retry_prompt = prompt

                for attempt in range(3):
                    candidate = (self.generate_text(retry_prompt) or "").strip()
                    candidate = " ".join(candidate.split())

                    if candidate and "?" not in candidate:
                        candidate = candidate + "?"

                    if candidate and not self._is_duplicate_question(candidate, previous_questions):
                        yield candidate
                        return

                    retry_prompt = f"""{prompt}
ADDITIONAL HARD CONSTRAINT:
- The next question MUST be semantically different from all previous assistant questions.
- Do NOT ask about age, gender, nausea, sunlight triggers, diet timing, food list, or time of day again.
- Ask about a new diagnostic axis: sleep, bowel habits, urine, stress, hydration, appetite, or relief factors.
- Output only one new question.
"""

                yield "How are your sleep quality, bowel habits, and hydration on days when the headache becomes worse?"
                return

            # Non-gathering modes keep streaming behavior.
            for response in self.client.models.generate_content_stream(
                model=self.model_id,
                contents=prompt,
                config={"temperature": config.TEMPERATURE}
            ):
                if response.text:
                    yield response.text
        except Exception as e:
            yield f"\n[Error during generation: {e}]"
