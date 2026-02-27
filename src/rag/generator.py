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
            # Get the source filename from the chunk metadata
            source = chunk.get("source", "Unknown")
            # Clean up source name (remove .pdf extension)
            if source.endswith(".pdf"):
                source = source.replace(".pdf", "")
            parts.append(f"Source {i+1} ({source}):\n{text}")
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
7. Ask questions from DIFFERENT books/sources - do not stick to only one book.

IMPORTANT: After your question, ADD the source in brackets. Show DIFFERENT sources each time:
Example: "What time of day do your headaches typically occur? (Ayurvedic-Home-Remedies-English)"
Next question could be: "What types of food do you typically consume? (Evidence_based_Ayurvedic_Practice)"

EXAMPLE OF WRONG OUTPUT: "Thank you for sharing that. Given your symptoms, I would like to ask..."
EXAMPLE OF CORRECT OUTPUT: "What time of day do your headaches typically occur? (ayurvedic_treatment_file1)"
"""
        elif mode == "diagnosis":
            instruction = """
OUTPUT FORMAT (STRICTLY FOLLOW THIS - ONE ITEM PER LINE):
BASE RULE: EACH OPTION MUST BE IN BULLET POINT FORMAT STARTING WITH THE EMOJI AND THE LABEL. DO NOT DEVIATE FROM THIS FORMAT.
All OF THEM NEEDS TO BE SEPERATED BY NEW LINES. DO NOT COMBINE MULTIPLE OPTIONS IN THE SAME LINE.
The output should be in bullet point format with the following sections:
--- USER-FRIENDLY OUTPUT ---

📋 DIAGNOSIS:
• [condition name - english and sanskrit if available]

📖 EXPLANATION:
• [5 bullet point explaining only]
• [Reason-1 from knowledge base with source in brackets]
• [Reason-2 from knowledge base with source in brackets]
• [Reason-3 from knowledge base with source in brackets]
• [Reason-4 from knowledge base with source in brackets]
• [Reason-5 from knowledge base with source in brackets]



⚠️ IS IT SERIOUS?
• [Yes or No followed by 3 bullet points explaining why - use sources in brackets]
• [Reason-1 from knowledge base with source in brackets]
• [Reason-2 from knowledge base with source in brackets]
• [Reason-3 from knowledge base with source in brackets]


🏠 CAN IT BE TREATED AT HOME?
• [Yes or No]
• [Brief reason followed by 3 bullet points explaining why - use sources in brackets]
• [Reason-1 from knowledge base with source in brackets]
• [Reason-2 from knowledge base with source in brackets]
• [Reason-3 from knowledge base with source in brackets]


Would you like home-based remedies, do's and don'ts, and lifestyle recommendations?

--- END USER OUTPUT ---
<--leave space between each content section and do not combine them into one line-->
*
*
*
*
*
<--IMPORTANT: When user responds to remedies question, ALWAYS provide remedies - do NOT ask more questions!-->
INTERNAL ANALYSIS (For your reference only):
Top 3 conditions in bullet point:
 * Disease Name - 1 (COMMON ENGLISH NAME AND AYURVEDIC NAME) [List with confidence scores] 
     -> Reason-1 from knowledge base with source in brackets
     -> Reason-2 from knowledge base with source in brackets

 * Disease Name - 2 (COMMON ENGLISH NAME AND AYURVEDIC NAME) [List with confidence scores]
        -> Reason-1 from knowledge base with source in brackets
        -> Reason-2 from knowledge base with source in brackets

* Disease Name - 3 (COMMON ENGLISH NAME AND AYURVEDIC NAME) [List with confidence scores]
        -> Reason-1 from knowledge base with source in brackets
        -> Reason-2 from knowledge base with source in brackets

Next ASK THE USER ONLY ONE QUESTION about remedies consent: "Would you like home-based remedies, do's and don'ts, and lifestyle recommendations?"
IMPORTANT: When user responds to remedies question, ALWAYS provide remedies - do NOT ask more questions!

STRICTLY use ONLY information from the RETRIEVED SOURCES. Use actual document names like "(ayurvedic_treatment_file1)", "(Ayurvedic-Home-Remedies-English)" for citations.
"""
        elif mode == "uncertain":
            instruction = """
OUTPUT FORMAT (STRICTLY FOLLOW THIS):
BASE RULE: EACH OPTION MUST BE IN BULLET POINT FORMAT STARTING WITH THE EMOJI AND THE LABEL. DO NOT DEVIATE FROM THIS FORMAT.
All OF THEM NEEDS TO BE SEPERATED BY NEW LINES. DO NOT COMBINE MULTIPLE OPTIONS IN THE SAME LINE.
The output should be in bullet point format with the following sections:

--- USER-FRIENDLY OUTPUT ---
[BULLET POINT FORMAT - Show this to user]

📋 DIAGNOSIS:
• [condition name - english and sanskrit if available]

📖 EXPLANATION:
• [5 bullet point explaining only]
• [Reason-1 from knowledge base with source in brackets]
• [Reason-2 from knowledge base with source in brackets]
• [Reason-3 from knowledge base with source in brackets]
• [Reason-4 from knowledge base with source in brackets]
• [Reason-5 from knowledge base with source in brackets]



⚠️ IS IT SERIOUS?
• [Yes or No followed by 3 bullet points explaining why - use sources in brackets]
• [Reason-1 from knowledge base with source in brackets]
• [Reason-2 from knowledge base with source in brackets]
• [Reason-3 from knowledge base with source in brackets]


🏠 CAN IT BE TREATED AT HOME?
• [Yes or No]
• [Brief reason followed by 3 bullet points explaining why - use sources in brackets]
• [Reason-1 from knowledge base with source in brackets]
• [Reason-2 from knowledge base with source in brackets]
• [Reason-3 from knowledge base with source in brackets]


Would you like home-based remedies, do's and don'ts, and lifestyle recommendations?
<--leave space between each content section and do not combine them into one line-->
*
*
*
*
*
<--IMPORTANT: When user responds to remedies question, ALWAYS provide remedies - do NOT ask more questions!-->
--- END USER OUTPUT ---

INTERNAL ANALYSIS (For your reference only):
Top 3 conditions in bullet point:
 * Disease Name - 1 (COMMON ENGLISH NAME AND AYURVEDIC NAME) [List with confidence scores] 
     -> Reason-1 from knowledge base with source in brackets
     -> Reason-2 from knowledge base with source in brackets

 * Disease Name - 2 (COMMON ENGLISH NAME AND AYURVEDIC NAME) [List with confidence scores]
        -> Reason-1 from knowledge base with source in brackets
        -> Reason-2 from knowledge base with source in brackets

* Disease Name - 3 (COMMON ENGLISH NAME AND AYURVEDIC NAME) [List with confidence scores]
        -> Reason-1 from knowledge base with source in brackets
        -> Reason-2 from knowledge base with source in brackets

Next ASK THE USER ONLY ONE QUESTION about remedies consent: "Would you like home-based remedies, do's and don'ts, and lifestyle recommendations?"
IMPORTANT: When user responds to remedies question, ALWAYS provide remedies - do NOT ask more questions!

IMPORTANT: When user responds to remedies question, ALWAYS provide remedies - do NOT ask more questions!

STRICTLY use ONLY information from the RETRIEVED SOURCES. Use actual document names like "(ayurvedic_treatment_file1)", "(Ayurvedic-Home-Remedies-English)" for citations.
"""
        elif mode == "uncertain_final":
            instruction = """
OUTPUT FORMAT (STRICTLY FOLLOW THIS):


--- USER-FRIENDLY OUTPUT ---
[BULLET POINT FORMAT - Show this to user]
BASE RULE: EACH OPTION MUST BE IN BULLET POINT FORMAT STARTING WITH THE EMOJI AND THE LABEL. DO NOT DEVIATE FROM THIS FORMAT.
All OF THEM NEEDS TO BE SEPERATED BY NEW LINES. DO NOT COMBINE MULTIPLE OPTIONS IN THE SAME LINE.
The output should be in bullet point format with the following sections:

📋 DIAGNOSIS:
• [condition name - english and sanskrit if available]

📖 EXPLANATION:
• [5 bullet point explaining only]
• [Reason-1 from knowledge base with source in brackets]
• [Reason-2 from knowledge base with source in brackets]
• [Reason-3 from knowledge base with source in brackets]
• [Reason-4 from knowledge base with source in brackets]
• [Reason-5 from knowledge base with source in brackets]



⚠️ IS IT SERIOUS?
• [Yes or No followed by 3 bullet points explaining why - use sources in brackets]
• [Reason-1 from knowledge base with source in brackets]
• [Reason-2 from knowledge base with source in brackets]
• [Reason-3 from knowledge base with source in brackets]


🏠 CAN IT BE TREATED AT HOME?
• [Yes or No]
• [Brief reason followed by 3 bullet points explaining why - use sources in brackets]
• [Reason-1 from knowledge base with source in brackets]
• [Reason-2 from knowledge base with source in brackets]
• [Reason-3 from knowledge base with source in brackets]


Would you like home-based remedies, do's and don'ts, and lifestyle recommendations?
<--leave space between each content section and do not combine them into one line-->
*
*
*
*
*
<--IMPORTANT: When user responds to remedies question, ALWAYS provide remedies - do NOT ask more questions!-->
--- END USER OUTPUT ---

INTERNAL ANALYSIS (For your reference only):
Top 3 conditions in bullet point:
 * Disease Name - 1 (COMMON ENGLISH NAME AND AYURVEDIC NAME) [List with confidence scores] 
     -> Reason-1 from knowledge base with source in brackets
     -> Reason-2 from knowledge base with source in brackets

 * Disease Name - 2 (COMMON ENGLISH NAME AND AYURVEDIC NAME) [List with confidence scores]
        -> Reason-1 from knowledge base with source in brackets
        -> Reason-2 from knowledge base with source in brackets

* Disease Name - 3 (COMMON ENGLISH NAME AND AYURVEDIC NAME) [List with confidence scores]
        -> Reason-1 from knowledge base with source in brackets
        -> Reason-2 from knowledge base with source in brackets

Next ASK THE USER ONLY ONE QUESTION about remedies consent: "Would you like home-based remedies, do's and don'ts, and lifestyle recommendations?"
IMPORTANT: When user responds to remedies question, ALWAYS provide remedies - do NOT ask more questions!

IMPORTANT: When user responds to remedies question, ALWAYS provide remedies - do NOT ask more questions!

STRICTLY use ONLY information from the RETRIEVED SOURCES. Use actual document names like "(ayurvedic_treatment_file1)", "(Ayurvedic-Home-Remedies-English)" for citations.
"""
        elif mode == "risk_gate_question":
            instruction = """
Ask exactly one safety question:
"Before I suggest remedies, are you currently on any medicines, or do you have liver disease, kidney disease, hypertension, or diabetes?"
Output only that question.
"""
        elif mode == "escalation":
            instruction = """
OUTPUT FORMAT (STRICTLY FOLLOW THIS):


--- USER-FRIENDLY OUTPUT ---
[BULLET POINT FORMAT - Show this to user]

📋 DIAGNOSIS:
• [condition name - english and sanskrit if available]

📖 EXPLANATION:
• [5 bullet point explaining only]
• [Reason-1 from knowledge base with source in brackets]
• [Reason-2 from knowledge base with source in brackets]
• [Reason-3 from knowledge base with source in brackets]
• [Reason-4 from knowledge base with source in brackets]
• [Reason-5 from knowledge base with source in brackets]



⚠️ IS IT SERIOUS?
• [Yes or No followed by 3 bullet points explaining why - use sources in brackets]
• [Reason-1 from knowledge base with source in brackets]
• [Reason-2 from knowledge base with source in brackets]
• [Reason-3 from knowledge base with source in brackets]


🏠 CAN IT BE TREATED AT HOME?
• [Yes or No]
• [Brief reason followed by 3 bullet points explaining why - use sources in brackets]
• [Reason-1 from knowledge base with source in brackets]
• [Reason-2 from knowledge base with source in brackets]
• [Reason-3 from knowledge base with source in brackets]


Would you like home-based remedies, do's and don'ts, and lifestyle recommendations?
<--leave space between each content section and do not combine them into one line-->
*
*
*
*
*
<--IMPORTANT: When user responds to remedies question, ALWAYS provide remedies - do NOT ask more questions!-->
--- END USER OUTPUT ---

INTERNAL ANALYSIS (For your reference only):
Top 3 conditions in bullet point:
 * Disease Name - 1 (COMMON ENGLISH NAME AND AYURVEDIC NAME) [List with confidence scores] 
     -> Reason-1 from knowledge base with source in brackets
     -> Reason-2 from knowledge base with source in brackets

 * Disease Name - 2 (COMMON ENGLISH NAME AND AYURVEDIC NAME) [List with confidence scores]
        -> Reason-1 from knowledge base with source in brackets
        -> Reason-2 from knowledge base with source in brackets

* Disease Name - 3 (COMMON ENGLISH NAME AND AYURVEDIC NAME) [List with confidence scores]
        -> Reason-1 from knowledge base with source in brackets
        -> Reason-2 from knowledge base with source in brackets

Next ASK THE USER ONLY ONE QUESTION about remedies consent: "Would you like home-based remedies, do's and don'ts, and lifestyle recommendations?"
IMPORTANT: When user responds to remedies question, ALWAYS provide remedies - do NOT ask more questions!

IMPORTANT: When user responds to remedies question, ALWAYS provide remedies - do NOT ask more questions!

STRICTLY use ONLY information from the RETRIEVED SOURCES. Use actual document names like "(ayurvedic_treatment_file1)", "(Ayurvedic-Home-Remedies-English)" for citations.
"""
        elif mode == "remedies":
            instruction = f"""
OUTPUT FORMAT (STRICTLY FOLLOW THIS - BULLET POINT FORMAT):

--- REMEDIES & LIFESTYLE ---
[BULLET POINT FORMAT - Show everything below to user]

🏠 HOME REMEDIES:
    - TOP 5 REMEDIES (WITH CITATIONS):
    • [Remedy 1 from knowledge base with source in brackets]
    • [Remedy 2 from knowledge base with source in brackets]
    • [Remedy 3 from knowledge base with source in brackets]
    • [Remedy 4 from knowledge base with source in brackets]
    • [Remedy 5 from knowledge base with source in brackets]
Additional Note on Remedies: [Any important notes on remedies based on the knowledge base]


✅ DO'S (THINGS YOU CAN HAVE/MUST DO):
-TOP 5 DO'S (WITH CITATIONS):
• [Do 1 from knowledge base with source in brackets]
• [Do 2 from knowledge base with source in brackets]
• [Do 3 from knowledge base with source in brackets]
• [Do 4 from knowledge base with source in brackets]
• [Do 5 from knowledge base with source in brackets]

❌ DON'TS (THINGS TO AVOID/MUST NOT DO):
-TOP 5 DON'TS (WITH CITATIONS):
• [Avoid 1 from knowledge base]
• [Avoid 2 from knowledge base with source in brackets]
• [Avoid 3 from knowledge base with source in brackets]
• [Avoid 4 from knowledge base with source in brackets]
• [Avoid 5 from knowledge base with source in brackets]

🍽️ FOOD TO HAVE:
• [Food 1 from knowledge base with source in brackets]
• [Food 2 from knowledge base with source in brackets]
• [Food 3 from knowledge base with source in brackets]
• [Food 1 from knowledge base with source in brackets]
• [Food 2 from knowledge base with source in brackets]
• [Food 3 from knowledge base with source in brackets]


🚫 FOOD TO AVOID:
• [Food 1 from knowledge base with source in brackets]
• [Food 2 from knowledge base with source in brackets]
• [Food 3 from knowledge base with source in brackets   ]

🌿 LIFESTYLE RECOMMENDATIONS:
• [Lifestyle tip 1 from knowledge base]
• [Lifestyle tip 2 from knowledge base]
• [Lifestyle tip 3 from knowledge base]

Additional Note on Lifestyle: [Any important notes on lifestyle based on the knowledge base]

--- END OUTPUT ---

Cite sources using actual document names like "(ayurvedic_treatment_file1)", "(Ayurvedic-Home-Remedies-English)".

STRICTLY use ONLY information from the RETRIEVED SOURCES. Do not hallucinate.
"""
        elif mode == "more_info":
            instruction = """
Provide more detailed information about the diagnosis.
Include:
- Detailed explanation of the condition
- Possible causes
- What to expect
- Any additional details from the knowledge base

Then ask: "Would you like home-based remedies, do's and don'ts, and lifestyle recommendations?"

Cite sources using actual document names like "(ayurvedic_treatment_file1)", "(Ayurvedic-Home-Remedies-English)".

STRICTLY use ONLY information from the RETRIEVED SOURCES.
"""
        else:
            instruction = "Provide a helpful response based on the RETRIEVED SOURCES. If providing medical information, cite sources."

        prompt = f"""
SYSTEM:
You are an Ayurvedic clinical assistant.

GLOBAL RULES (APPLY TO ALL MODES):
- NO sympathy statements (e.g., "I understand this must be difficult")
- NO acknowledgments (e.g., "Thank you for sharing", "I appreciate that information")
- NO conversational filler (e.g., "It's interesting that...", "Given what you've told me...")
- Go directly to the content requested
- ALWAYS cite sources when providing medical information (e.g., "(Source 1)", "(Ayurvedic-Home-Remedies-English)")
- Use information ONLY from the RETRIEVED SOURCES

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

            # Non-gathering modes - use non-streaming for better structure
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config={"temperature": config.TEMPERATURE}
            )
            output = response.text or ""
            # Clean the output to remove duplicates
            output = self._clean_output(output)
            yield output
        except Exception as e:
            yield f"\n[Error during generation: {e}]"

    def _clean_output(self, text: str) -> str:
        """
        Cleans repetition, duplicated lines,
        and structured section looping.
        """
        if not text:
            return text

        lines = text.splitlines()
        cleaned_lines = []
        seen_lines = set()

        for line in lines:
            normalized = line.strip()

            # Remove exact duplicate lines
            if normalized and normalized in seen_lines:
                continue

            cleaned_lines.append(line)
            seen_lines.add(normalized)

        cleaned_text = "\n".join(cleaned_lines)

        cleaned_text = self._dedupe_sections(cleaned_text)

        return cleaned_text

    def _dedupe_sections(self, text: str) -> str:
        """
        Prevent repeated section headers
        like multiple FOOD / DO'S / REMEDIES blocks.
        """

        section_markers = [
            "HOME REMEDIES",
            "DO'S",
            "DON'TS",
            "DIET",
            "LIFESTYLE",
            "FOOD",
        ]

        for marker in section_markers:
            count = text.count(marker)
            if count > 1:
                first_index = text.find(marker)
                text = (
                    text[:first_index]
                    + text[first_index:].replace(marker, "", count - 1)
                )

        return text
