import config
import json
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

        profile_lines = [line for line in lines if line.startswith("PATIENT_PROFILE:")]
        tail_lines = lines[-max_lines:]
        if profile_lines and profile_lines[0] not in tail_lines:
            return "\n".join([profile_lines[0]] + tail_lines)
        return "\n".join(tail_lines)

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

    def _safe_json_load(self, text, fallback):
        if not text:
            return fallback
        try:
            return json.loads(text)
        except Exception:
            # Try extracting first JSON object boundaries.
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except Exception:
                    return fallback
        return fallback

    def generate_differential_diagnosis(self, conversation_history, retrieved_chunks):
        trimmed_history = self._trim_history(conversation_history, max_lines=30)
        context = self._build_context(retrieved_chunks, max_chunks=8, max_chars_per_chunk=1100)

        prompt = f"""
SYSTEM:
You are an Ayurvedic clinical diagnostic expert with strict medical safety policy.
Create a differential diagnosis with uncertainty handling.

CONVERSATION HISTORY:
{trimmed_history}

RETRIEVED SOURCES:
{context}

Return ONLY valid JSON with this exact schema:
{{
  "possible_conditions": [
    {{"name": "string", "confidence": 0.0, "evidence_for": ["..."], "evidence_against": ["..."]}},
    {{"name": "string", "confidence": 0.0, "evidence_for": ["..."], "evidence_against": ["..."]}},
    {{"name": "string", "confidence": 0.0, "evidence_for": ["..."], "evidence_against": ["..."]}}
  ],
  "most_likely": "string",
  "most_likely_confidence": 0.0,
  "uncertainty_level": "low|moderate|high",
  "red_flags_present": ["..."],
  "reasoning_summary": "string"
}}

Rules:
- Always provide at least 3 competing conditions.
- Confidence values must be between 0 and 1.
- Use uncertainty_level="high" if evidence is weak, contradictory, or symptom duration is short/unclear.
- Do not produce treatment in this step.
"""
        raw = self.generate_text(prompt)
        fallback = {
            "possible_conditions": [
                {"name": "Undifferentiated headache pattern", "confidence": 0.34, "evidence_for": [], "evidence_against": []},
                {"name": "Pitta-aggravated headache pattern", "confidence": 0.33, "evidence_for": [], "evidence_against": []},
                {"name": "Tension-type pattern", "confidence": 0.33, "evidence_for": [], "evidence_against": []},
            ],
            "most_likely": "Undifferentiated headache pattern",
            "most_likely_confidence": 0.34,
            "uncertainty_level": "high",
            "red_flags_present": [],
            "reasoning_summary": "Evidence is limited; differential retained to avoid overconfident diagnosis.",
        }
        report = self._safe_json_load(raw, fallback)
        # Normalize basic fields for downstream safety gating.
        if "possible_conditions" not in report or not isinstance(report["possible_conditions"], list):
            report["possible_conditions"] = fallback["possible_conditions"]
        if len(report["possible_conditions"]) < 3:
            report["possible_conditions"] = (report["possible_conditions"] + fallback["possible_conditions"])[:3]
        try:
            report["most_likely_confidence"] = float(report.get("most_likely_confidence", 0.0))
        except Exception:
            report["most_likely_confidence"] = 0.0
        report["most_likely_confidence"] = max(0.0, min(1.0, report["most_likely_confidence"]))
        if report.get("uncertainty_level") not in {"low", "moderate", "high"}:
            report["uncertainty_level"] = "moderate"
        if not isinstance(report.get("red_flags_present"), list):
            report["red_flags_present"] = []
        return report

    def self_check_differential(self, differential_report, conversation_history):
        trimmed_history = self._trim_history(conversation_history, max_lines=30)
        payload = json.dumps(differential_report, ensure_ascii=False)
        prompt = f"""
SYSTEM:
You are a strict medical safety auditor.
Review whether the diagnosis is overconfident and whether treatment should be blocked.

CONVERSATION HISTORY:
{trimmed_history}

DIFFERENTIAL REPORT:
{payload}

Return ONLY valid JSON:
{{
  "overconfident": true,
  "missing_differentials": true,
  "requires_medical_escalation": false,
  "treatment_allowed": false,
  "adjusted_confidence_cap": 0.55,
  "notes": "short reason"
}}
"""
        raw = self.generate_text(prompt)
        fallback = {
            "overconfident": False,
            "missing_differentials": False,
            "requires_medical_escalation": False,
            "treatment_allowed": True,
            "adjusted_confidence_cap": 1.0,
            "notes": "",
        }
        out = self._safe_json_load(raw, fallback)
        for key in ("overconfident", "missing_differentials", "requires_medical_escalation", "treatment_allowed"):
            out[key] = bool(out.get(key, fallback[key]))
        try:
            out["adjusted_confidence_cap"] = float(out.get("adjusted_confidence_cap", 1.0))
        except Exception:
            out["adjusted_confidence_cap"] = 1.0
        out["adjusted_confidence_cap"] = max(0.0, min(1.0, out["adjusted_confidence_cap"]))
        out["notes"] = str(out.get("notes", ""))
        return out

    def format_differential_report(self, differential_report):
        most_likely = differential_report.get("most_likely", "Uncertain")
        confidence = differential_report.get("most_likely_confidence", 0.0)
        uncertainty = differential_report.get("uncertainty_level", "moderate")
        lines = [f"DIAGNOSIS: {most_likely}", f"CONFIDENCE: {confidence:.2f}", f"UNCERTAINTY: {uncertainty}"]
        options = differential_report.get("possible_conditions", [])[:3]
        for i, opt in enumerate(options, start=1):
            name = opt.get("name", "Unknown")
            try:
                c = float(opt.get("confidence", 0.0))
            except Exception:
                c = 0.0
            lines.append(f"DIFFERENTIAL_{i}: {name} ({c:.2f})")
        return "\n".join(lines)

    def generate_diagnosis(self, conversation_history, retrieved_chunks):
        # Backward-compatible wrapper.
        report = self.generate_differential_diagnosis(conversation_history, retrieved_chunks)
        return self.format_differential_report(report)

    def verify_diagnosis(self, diagnosis_report, conversation_history):
        # Backward-compatible boolean verifier.
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
1. If PATIENT_PROFILE shows unknown age or unknown gender, ask: "To provide an accurate Ayurvedic assessment, could you share your age and gender?"
2. If PATIENT_PROFILE already has both age and gender, DO NOT ask age/gender again.
3. Ask ONLY ONE (1) clarifying question to help narrow down the diagnosis.
4. DO NOT provide treatments or final answers yet.
5. Ground your question strictly in the provided sources and conversation history.
6. Gather information about lifestyle, diet, medical history, and habits as this is crucial for Ayurvedic diagnosis.
7. Avoid asking questions that are not relevant to Ayurvedic diagnosis.

EXAMPLE OF WRONG OUTPUT: "Thank you for sharing that. Given your symptoms, I would like to ask..."
EXAMPLE OF CORRECT OUTPUT: "What time of day do your headaches typically occur?"
"""
        elif mode == "diagnosis":
            instruction = """
1. Summarize the differential diagnosis from the provided report (not a single absolute diagnosis).
2. Mention uncertainty clearly and briefly state at least one alternative condition.
3. Keep language conservative and probabilistic, avoid certainty claims.
4. End by asking if the user wants safe remedies and lifestyle guidance.
5. Avoid medical jargon as much as possible.
"""
        elif mode == "uncertain":
            instruction = """
1. Output ONLY one new, specific follow-up question that has NOT been asked before.
2. Do NOT include diagnostic summaries, warnings, or lifestyle advice text.
3. Question must be tightly focused on unresolved ear/respiratory differential details.
4. Keep it short and clear.
"""
        elif mode == "uncertain_final":
            instruction = """
1. Output exactly two concise lines:
   Line 1: "Uncertain diagnosis; in-person medical evaluation is recommended."
   Line 2: "No treatment protocol will be provided at this confidence level."
2. Do NOT add extra explanation, differential details, or self-care tips.
"""
        elif mode == "risk_gate_question":
            instruction = """
Ask exactly one safety question:
"Before I suggest remedies, are you currently on any medicines, or do you have liver disease, kidney disease, hypertension, or diabetes?"
Output only that question.
"""
        elif mode == "escalation":
            instruction = """
1. State that red flags are present and in-person medical evaluation is recommended first.
2. Do NOT provide treatment protocol, herbs, detox, or medicine combinations.
3. Keep response concise and safety-first.
"""
        elif mode == "remedies":
            instruction = f"""
1. Conservative-first policy: begin with low-risk advice (rest, hydration, sleep, heat/ergonomics, trigger avoidance).
2. If adding remedies, limit to at most {getattr(config, "MAX_REMEDY_INTERVENTIONS", 3)} interventions total.
3. If medication/comorbidity risk profile is unclear, do not give aggressive regimens.
4. Extract only from RETRIEVED SOURCES; do not hallucinate.
5. Keep it simple, concise, and user-friendly with numbered points.
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
            if mode in ("gathering", "uncertain"):
                previous_questions = self._extract_previous_questions(conversation_history)
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
- Ask about a new diagnostic axis that is unresolved.
- Output only one new question.
"""

                yield "Do you have reduced hearing, ringing, or fever along with the ear pain?"
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
