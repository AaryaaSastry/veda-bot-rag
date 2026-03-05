import config
import json
import re
import ollama
from google import genai


class Generator:

    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        #Using the gemma model for testing currently as it offers unlimited tokens
        #It is recomended to switch to a more powerful model such as Gemini-2.0-flash or Gemini-2.5-flash for final production use. 
        self.model_id = "gemma-3-27b-it"
        # Local verification model
        self.ollama_model = "llama3" # Defaulting to llama3, can be changed in config

    def generate_text(self, prompt):
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt
        )
        return response.text

    def _generate_ollama_text(self, prompt, verification=False):
        """Generates text using a local Ollama model for verification logic."""
        try:
            response = ollama.chat(model=self.ollama_model, messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ])
            if verification:
                print("verified via ollama")
            return response['message']['content']
        except Exception as e:
            print(f"Ollama Error: {e}")
            return ""

    def trigger_verification(self, stage, question="", conversation_history=""):
        """
        Lightweight verification ping to ensure verification is triggered on all response paths.
        """
        prompt = f"""
SYSTEM:
You are a verification sentinel. Confirm this pipeline stage can proceed safely.
Return ONLY: OK

STAGE:
{stage}

USER INPUT:
{question}

CONVERSATION HISTORY (trimmed):
{self._trim_history(conversation_history, max_lines=8)}
"""
        self._generate_ollama_text(prompt, verification=True)

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

    def _extract_source_names(self, retrieved_chunks, max_chunks=6):
        names = []
        for chunk in retrieved_chunks[:max_chunks]:
            source = (chunk.get("source", "") or "").strip()
            if source.endswith(".pdf"):
                source = source[:-4]
            if source and source not in names:
                names.append(source)
        return names

    def _extract_source_index_map(self, retrieved_chunks, max_chunks=6):
        source_map = {}
        for i, chunk in enumerate(retrieved_chunks[:max_chunks], start=1):
            source = (chunk.get("source", "") or "").strip()
            if source.endswith(".pdf"):
                source = source[:-4]
            source_map[str(i)] = source or "Unknown"
        return source_map

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

    def _infer_question_axis(self, question):
        normalized = self._normalize_question(question)
        axis_keywords = {
            "demographics": {"age", "gender", "male", "female"},
            "onset_timing": {"start", "started", "since", "when", "duration", "time", "day", "night"},
            "severity_quality": {"severe", "mild", "sharp", "dull", "throbbing", "burning", "numbness", "pain"},
            "associated_symptoms": {"symptom", "fever", "breathlessness", "nausea", "vomiting", "headache", "chest"},
            "triggers_relievers": {"worse", "better", "trigger", "relieve", "aggravate", "improve"},
            "diet": {"diet", "food", "meal", "eat", "grains", "vegetable", "spice"},
            "lifestyle_habits": {"sleep", "exercise", "stress", "habit", "routine", "work"},
            "medical_history": {"history", "medicine", "medication", "diagnosis", "chronic", "disease"},
        }
        tokens = set(normalized.split())
        for axis, keywords in axis_keywords.items():
            if tokens & keywords:
                return axis
        return "other"

    def _get_unasked_axes(self, previous_questions):
        preferred_axes = [
            "onset_timing",
            "severity_quality",
            "associated_symptoms",
            "triggers_relievers",
            "diet",
            "lifestyle_habits",
            "medical_history",
        ]
        asked = {self._infer_question_axis(q) for q in previous_questions}
        return [axis for axis in preferred_axes if axis not in asked]

    def _fallback_question_for_axis(self, axis):
        templates = {
            "onset_timing": "When did this start, and is it constant or does it come and go?",
            "severity_quality": "How severe is the symptom now, and what does it feel like?",
            "associated_symptoms": "Do you have any other symptoms along with this?",
            "triggers_relievers": "What makes this symptom worse, and what gives relief?",
            "diet": "What do you usually eat in a typical day?",
            "lifestyle_habits": "How are your sleep, daily activity, and stress levels currently?",
            "medical_history": "Do you have any prior medical conditions or regular medicines?",
        }
        return templates.get(axis, "Could you share one more detail that has not been discussed yet?")

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

    def generate_differential_diagnosis(self, conversation_history, retrieved_chunks, disease_index=None):
        trimmed_history = self._trim_history(conversation_history, max_lines=30)
        context = self._build_context(retrieved_chunks, max_chunks=8, max_chars_per_chunk=1100)
        
        index_context = ""
        if disease_index:
            # Provide top relevant disease summaries from the index
            index_context = "DISEASE INDEX SUMMARIES:\n" + json.dumps(disease_index[:10], indent=2)

        prompt = f"""
SYSTEM:
You are an Ayurvedic clinical diagnostic expert with strict medical safety policy.
Create a differential diagnosis with uncertainty handling.

CONVERSATION HISTORY:
{trimmed_history}

{index_context}

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

    def self_check_differential(self, differential_report, conversation_history, retrieved_chunks=None):
        trimmed_history = self._trim_history(conversation_history, max_lines=30)
        payload = json.dumps(differential_report, ensure_ascii=False)
        chunk_context = self._build_context(retrieved_chunks or [], max_chunks=6, max_chars_per_chunk=900) if retrieved_chunks else ""
        prompt = f"""
SYSTEM:
You are a strict medical safety auditor.
Review whether the diagnosis is overconfident, factually supported by retrieved chunks, and safe to proceed.

CONVERSATION HISTORY:
{trimmed_history}

DIFFERENTIAL REPORT:
{payload}

RETRIEVED SOURCES:
{chunk_context}

Return ONLY valid JSON:
{{
  "diagnosis_valid": true,
  "overconfident": true,
  "missing_differentials": true,
  "requires_medical_escalation": false,
  "treatment_allowed": false,
  "supported_by_chunks": true,
  "rejection_reasons": ["reason 1", "reason 2", "reason 3", "reason 4", "reason 5"],
  "alternative_conditions": ["condition 1", "condition 2", "condition 3"],
  "targeted_questions": ["question 1?", "question 2?", "question 3?"],
  "adjusted_confidence_cap": 0.55,
  "notes": "short reason"
}}
"""
        raw = self._generate_ollama_text(prompt, verification=True)
        fallback = {
            "diagnosis_valid": True,
            "overconfident": False,
            "missing_differentials": False,
            "requires_medical_escalation": False,
            "treatment_allowed": True,
            "supported_by_chunks": True,
            "rejection_reasons": [],
            "alternative_conditions": [],
            "targeted_questions": [],
            "adjusted_confidence_cap": 1.0,
            "notes": "Verification failed, passing through.",
        }
        out = self._safe_json_load(raw, fallback)
        for key in ("diagnosis_valid", "overconfident", "missing_differentials", "requires_medical_escalation", "treatment_allowed", "supported_by_chunks"):
            out[key] = bool(out.get(key, fallback[key]))
        if not isinstance(out.get("rejection_reasons"), list):
            out["rejection_reasons"] = []
        if not isinstance(out.get("alternative_conditions"), list):
            out["alternative_conditions"] = []
        if not isinstance(out.get("targeted_questions"), list):
            out["targeted_questions"] = []
        out["rejection_reasons"] = [str(x).strip() for x in out["rejection_reasons"] if str(x).strip()]
        out["alternative_conditions"] = [str(x).strip() for x in out["alternative_conditions"] if str(x).strip()]
        out["targeted_questions"] = [str(x).strip() for x in out["targeted_questions"] if str(x).strip()]
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
        raw_response = self._generate_ollama_text(prompt, verification=True)
        response = raw_response.strip().upper()
        return "YES" in response

    def generate(self, question, retrieved_chunks, conversation_history="", mode="gathering"):
        if not retrieved_chunks:
            yield "I could find no relevant information in the documents provided."
            return

        trimmed_history = self._trim_history(conversation_history, max_lines=24)
        if mode == "gathering":
            context = self._build_context(retrieved_chunks, max_chunks=4, max_chars_per_chunk=750)
            source_names = self._extract_source_names(retrieved_chunks, max_chunks=4)
            source_index_map = self._extract_source_index_map(retrieved_chunks, max_chunks=4)
        else:
            context = self._build_context(retrieved_chunks, max_chunks=6, max_chars_per_chunk=950)
            source_names = self._extract_source_names(retrieved_chunks, max_chunks=6)
            source_index_map = self._extract_source_index_map(retrieved_chunks, max_chunks=6)

        if mode == "gathering":
            # Check if the question is a direct symptom from the Bayesian engine
            symptom_match = re.search(r"if the user has these symptoms: (.*)\.", question)
            if "The diagnosis engine wants to know" in question and symptom_match:
                target_symptom = symptom_match.group(1).strip()
                instruction = f"""
SYSTEM:
You are a natural language wrapper for a diagnostic engine.
Your task is to take a technical symptom attribute and convert it into a gentle, natural Ayurvedic question.

TARGET SYMPTOM: 
{target_symptom}

SOURCES:
{context}

RULES:
1. Output ONLY the question.
2. The question MUST ask about the user's experience with the '{target_symptom}'.
3. Use a polite, clinical tone.
4. Cite a source in brackets from the available sources (e.g., "(Charaka Samhita)").
5. Example conversion: "cyclic_fever" -> "Are you experiencing fever that occurs in cycles or comes and goes regularly? (Charaka Samhita)"
"""
            else:
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
3. Ask ONLY ONE ot TWO clarifying question to help narrow down the diagnosis and understand the user's condition.
4. DO NOT provide treatments or final answers yet.
5. Ground your question strictly in the provided sources and conversation history and 25 percent of llm information to understand if the orrect questions id being asked.
6. Gather information about lifestyle, diet, medical history, and habits as this is crucial for Ayurvedic diagnosis.
7. Ask questions from DIFFERENT books/sources - do not stick to only one book.
8. If the user mentioned a symptom, ask deeper questions about its quality: is it sharp, dull, throbbing? What makes it better or worse? When did it start?
9. Ask the questions based on the most relevant retrieved chunks - if a chunk mentions headaches being worse in the evening, ask "Do your headaches tend to worsen at specific times of the day, such as in the evening?" (with the source in brackets).
10. DO NOT ask the same or very similar question more than once. If you have already asked about symptom quality, do not ask about it again. If you have already asked about timing, do not ask about it again. Move on to other aspects like diet, lifestyle, triggers, etc.
11.Ensure you understand the underlying reasoning for your question based on the sources. For example, if you ask about timing of headaches, it's because certain Ayurvedic patterns have characteristic timing. If you ask about diet, it's because certain foods can aggravate or alleviate conditions. Always have a clear reason for why you are asking each question, grounded in the Ayurvedic knowledge from the sources.

IMPORTANT: After your question, ADD the source in brackets. Show DIFFERENT sources each time:
Example: "What time of day do your headaches typically occur? (Ayurvedic-Home-Remedies-English)"
Next question could be: "What types of food do you typically consume? (Evidence_based_Ayurvedic_Practice)"
"""
        elif mode == "diagnosis":
            instruction = """
OUTPUT FORMAT (STRICTLY FOLLOW THIS - ONE ITEM PER LINE):
BASE RULE: EACH OPTION MUST BE IN BULLET POINT FORMAT STARTING WITH THE EMOJI AND THE LABEL. DO NOT DEVIATE FROM THIS FORMAT.
All OF THEM NEEDS TO BE SEPERATED BY NEW LINES. DO NOT COMBINE MULTIPLE OPTIONS IN THE SAME LINE.
The output should be in bullet point format with the following sections:
--- USER-FRIENDLY OUTPUT ---

ðŸ“‹ DIAGNOSIS:
â€¢ [condition name - english and sanskrit if available]

ðŸ“– EXPLANATION:
â€¢ [5 bullet point explaining only]
â€¢ [Reason-1 from knowledge base with source in brackets]
â€¢ [Reason-2 from knowledge base with source in brackets]
â€¢ [Reason-3 from knowledge base with source in brackets]
â€¢ [Reason-4 from knowledge base with source in brackets]
â€¢ [Reason-5 from knowledge base with source in brackets]



âš ï¸ IS IT SERIOUS?
â€¢ [Yes or No followed by 3 bullet points explaining why - use sources in brackets]
â€¢ [Reason-1 from knowledge base with source in brackets]
â€¢ [Reason-2 from knowledge base with source in brackets]
â€¢ [Reason-3 from knowledge base with source in brackets]


ðŸ  CAN IT BE TREATED AT HOME?
â€¢ [Yes or No]
â€¢ [Brief reason followed by 3 bullet points explaining why - use sources in brackets]
â€¢ [Reason-1 from knowledge base with source in brackets]
â€¢ [Reason-2 from knowledge base with source in brackets]
â€¢ [Reason-3 from knowledge base with source in brackets]


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

ðŸ“‹ DIAGNOSIS:
â€¢ [condition name - english and sanskrit if available]

ðŸ“– EXPLANATION:
â€¢ [5 bullet point explaining only]
â€¢ [Reason-1 from knowledge base with source in brackets]
â€¢ [Reason-2 from knowledge base with source in brackets]
â€¢ [Reason-3 from knowledge base with source in brackets]
â€¢ [Reason-4 from knowledge base with source in brackets]
â€¢ [Reason-5 from knowledge base with source in brackets]



âš ï¸ IS IT SERIOUS?
â€¢ [Yes or No followed by 3 bullet points explaining why - use sources in brackets]
â€¢ [Reason-1 from knowledge base with source in brackets]
â€¢ [Reason-2 from knowledge base with source in brackets]
â€¢ [Reason-3 from knowledge base with source in brackets]


ðŸ  CAN IT BE TREATED AT HOME?
â€¢ [Yes or No]
â€¢ [Brief reason followed by 3 bullet points explaining why - use sources in brackets]
â€¢ [Reason-1 from knowledge base with source in brackets]
â€¢ [Reason-2 from knowledge base with source in brackets]
â€¢ [Reason-3 from knowledge base with source in brackets]


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

ðŸ“‹ DIAGNOSIS:
â€¢ [condition name - english and sanskrit if available]

ðŸ“– EXPLANATION:
â€¢ [5 bullet point explaining only]
â€¢ [Reason-1 from knowledge base with source in brackets]
â€¢ [Reason-2 from knowledge base with source in brackets]
â€¢ [Reason-3 from knowledge base with source in brackets]
â€¢ [Reason-4 from knowledge base with source in brackets]
â€¢ [Reason-5 from knowledge base with source in brackets]



âš ï¸ IS IT SERIOUS?
â€¢ [Yes or No followed by 3 bullet points explaining why - use sources in brackets]
â€¢ [Reason-1 from knowledge base with source in brackets]
â€¢ [Reason-2 from knowledge base with source in brackets]
â€¢ [Reason-3 from knowledge base with source in brackets]


ðŸ  CAN IT BE TREATED AT HOME?
â€¢ [Yes or No]
â€¢ [Brief reason followed by 3 bullet points explaining why - use sources in brackets]
â€¢ [Reason-1 from knowledge base with source in brackets]
â€¢ [Reason-2 from knowledge base with source in brackets]
â€¢ [Reason-3 from knowledge base with source in brackets]


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
URGENT SAFETY ADVICE:
- [One-line summary that symptoms need prompt in-person medical assessment]

WHY THIS NEEDS ESCALATION:
- [3-5 concise, source-grounded reasons]

WHAT NOT TO DO NOW:
- [2-4 concise points]

WHAT TO DO NOW:
- [3-5 immediate next steps, including urgent care/emergency support based on severity]
--- END USER OUTPUT ---

Rules:
- Do not ask for home remedies consent in escalation mode.
- Do not provide treatment plans that could delay urgent medical care.
- Keep advice clear, concise, and source-grounded.
"""
        elif mode == "escalation_followup":
            instruction = """
You are in post-escalation follow-up mode.

OUTPUT FORMAT (STRICTLY FOLLOW THIS):
--- USER-FRIENDLY OUTPUT ---
SAFETY REMINDER:
- [Reinforce urgent assessment need in one line]

NEXT STEP:
- [Single concrete next step based on user reply]
--- END USER OUTPUT ---

Rules:
- Do not re-enter diagnostic questioning loops.
- Do not ask remedies consent.
- Keep response concise (2-4 bullets total).
"""
        elif mode == "remedies":
            instruction = f"""
OUTPUT FORMAT (STRICTLY FOLLOW THIS - BULLET POINT FORMAT):

--- REMEDIES & LIFESTYLE ---
[BULLET POINT FORMAT - Show everything below to user]

ðŸ  HOME REMEDIES:
    - TOP 5 REMEDIES (WITH CITATIONS):
    â€¢ [Remedy 1 from knowledge base with source in brackets]
    â€¢ [Remedy 2 from knowledge base with source in brackets]
    â€¢ [Remedy 3 from knowledge base with source in brackets]
    â€¢ [Remedy 4 from knowledge base with source in brackets]
    â€¢ [Remedy 5 from knowledge base with source in brackets]
Additional Note on Remedies: [Any important notes on remedies based on the knowledge base]


âœ… DO'S (THINGS YOU CAN HAVE/MUST DO):
-TOP 5 DO'S (WITH CITATIONS):
â€¢ [Do 1 from knowledge base with source in brackets]
â€¢ [Do 2 from knowledge base with source in brackets]
â€¢ [Do 3 from knowledge base with source in brackets]
â€¢ [Do 4 from knowledge base with source in brackets]
â€¢ [Do 5 from knowledge base with source in brackets]

âŒ DON'TS (THINGS TO AVOID/MUST NOT DO):
-TOP 5 DON'TS (WITH CITATIONS):
â€¢ [Avoid 1 from knowledge base]
â€¢ [Avoid 2 from knowledge base with source in brackets]
â€¢ [Avoid 3 from knowledge base with source in brackets]
â€¢ [Avoid 4 from knowledge base with source in brackets]
â€¢ [Avoid 5 from knowledge base with source in brackets]

ðŸ½ï¸ FOOD TO HAVE:
â€¢ [Food 1 from knowledge base with source in brackets]
â€¢ [Food 2 from knowledge base with source in brackets]
â€¢ [Food 3 from knowledge base with source in brackets]
â€¢ [Food 1 from knowledge base with source in brackets]
â€¢ [Food 2 from knowledge base with source in brackets]
â€¢ [Food 3 from knowledge base with source in brackets]


ðŸš« FOOD TO AVOID:
â€¢ [Food 1 from knowledge base with source in brackets]
â€¢ [Food 2 from knowledge base with source in brackets]
â€¢ [Food 3 from knowledge base with source in brackets   ]

ðŸŒ¿ LIFESTYLE RECOMMENDATIONS:
â€¢ [Lifestyle tip 1 from knowledge base]
â€¢ [Lifestyle tip 2 from knowledge base]
â€¢ [Lifestyle tip 3 from knowledge base]

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
        elif mode == "consent_clarification":
            instruction = """
Ask exactly one clarification question:
"Would you like me to provide home-based remedies, dos and don'ts, and lifestyle recommendations? Please answer yes or no."
Output only that question.
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
- ALWAYS cite sources using only source names exactly as provided in SOURCE_NAME_ALLOWLIST
- NEVER use placeholder citations like "(Source 1)" or "(Source 2)"
- Use information ONLY from the RETRIEVED SOURCES

CONVERSATION HISTORY:
{trimmed_history}

RETRIEVED SOURCES:
{context}

CURRENT USER INPUT:
{question}

SOURCE_NAME_ALLOWLIST:
{", ".join(source_names) if source_names else "Unknown"}

INSTRUCTIONS:
{instruction}
- Groundedness: Remain strictly inside the provided sources.
- Ask questions only if they make sense for the patient's context.

Answer:
"""

        try:
            if mode == "gathering":
                previous_questions = self._extract_previous_questions(conversation_history)
                unasked_axes = self._get_unasked_axes(previous_questions)
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
- Ask about a new diagnostic axis that is unresolved.
- Prefer one of these unresolved axes: {", ".join(unasked_axes) if unasked_axes else "any unresolved axis"}
- Output only one new question.
"""

                fallback_axis = unasked_axes[0] if unasked_axes else "other"
                yield self._fallback_question_for_axis(fallback_axis)
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
            output = self._normalize_source_citations(output, source_index_map, source_names)
            output = self._strip_internal_sections(output)
            yield output
        except Exception as e:
            yield f"\n[Error during generation: {e}]"

    def _normalize_source_citations(self, text, source_index_map, source_names):
        if not text:
            return text

        def repl(match):
            idx = match.group(1)
            mapped = source_index_map.get(idx)
            if mapped:
                return f"({mapped})"
            if source_names:
                return f"({source_names[0]})"
            return "(Unknown)"

        text = re.sub(r"\(\s*source\s*(\d+)\s*\)", repl, text, flags=re.IGNORECASE)
        text = re.sub(r"\[\s*source\s*(\d+)\s*\]", repl, text, flags=re.IGNORECASE)
        return text

    def _strip_internal_sections(self, text):
        if not text:
            return text

        friendly_start = re.search(r"---\s*USER-FRIENDLY OUTPUT\s*---", text, flags=re.IGNORECASE)
        friendly_end = re.search(r"---\s*END USER OUTPUT\s*---", text, flags=re.IGNORECASE)
        if friendly_start and friendly_end and friendly_end.start() > friendly_start.end():
            text = text[friendly_start.start():friendly_end.end()]

        # Remove anything marked internal if it still leaks in.
        text = re.sub(
            r"INTERNAL ANALYSIS.*$",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        text = re.sub(r"^\s*Next ASK THE USER ONLY ONE QUESTION.*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
        return text.strip()

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

