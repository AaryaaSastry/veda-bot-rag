import config
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

    def generate_diagnosis(self, conversation_history, retrieved_chunks):
        context = "\n\n".join([f"Source {i+1}:\n{chunk['text']}" for i, chunk in enumerate(retrieved_chunks)])
        prompt = f"""
SYSTEM:
You are an Ayurvedic clinical diagnostic expert.
Based ON THE RETRIEVED SOURCES and the conversation history, provide a potential diagnosis and the reasoning behind it.

CONVERSATION HISTORY:
{conversation_history}

RETRIEVED SOURCES:
{context}

Format your output exactly like this:
DIAGNOSIS: [Name of condition]
REASONING: [Step by step reasoning based on symptoms and sources]
"""
        return self.generate_text(prompt)

    def verify_diagnosis(self, diagnosis_report, conversation_history):
        prompt = f"""
SYSTEM:
You are a senior Ayurvedic medical reviewer.
Look at the following diagnosis, reasoning, and the full conversation history.
Ensure that you understand the age and gender of the patient from the conversation history and use that for further diagnosis.
CONVERSATION HISTORY:
{conversation_history}

DIAGNOSIS REPORT:
{diagnosis_report}

Does this diagnosis make sense and is it sufficiently supported by the facts gathered from the user?
Respond with ONLY one word: "YES" or "NO".
"""
        response = self.generate_text(prompt).strip().upper()
        return "YES" in response

    def generate(self, question, retrieved_chunks, conversation_history="", mode="gathering"):
        if not retrieved_chunks:
            yield "I could find no relevant information in the documents provided."
            return

        context = "\n\n".join(
            [f"Source {i+1}:\n{chunk['text']}"
             for i, chunk in enumerate(retrieved_chunks)]
        )

        if mode == "gathering":
            instruction = """
STRICT OUTPUT RULES:
1. Output ONLY the question. Nothing else. No acknowledgments, no sympathy, no conversational filler.
2. DO NOT say things like "Thank you for providing your age and gender" or "It's interesting that..." or any similar statements.
3. DO NOT provide any context, explanations, or commentary before or after the question.
4. Start your response directly with the question word (What, How, When, Where, Which, etc.) or a direct question.

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
{conversation_history}

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
            # Using generate_content_stream for real-time response
            for response in self.client.models.generate_content_stream(
                model=self.model_id,
                contents=prompt,
                config={
                    "temperature": config.TEMPERATURE,
                }
            ):
                if response.text:
                    yield response.text
        except Exception as e:
            yield f"\n[Error during generation: {e}]"
