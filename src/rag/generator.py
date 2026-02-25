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
BASIC RULE FOR ALL MODES: NO CITATIONS TO BE PROVIDED IN THE ANSWERS AND QUESTIONS. 
BASE RULE: Ensure that all the questions asked are with the goal of narrowing down to a diagnosis in mind.

CRITICAL FIRST STEP: Check the conversation history. If age and gender are NOT mentioned yet, ask for BOTH in your first response naturally (e.g., "To help me provide an accurate Ayurvedic assessment, could you share your age and gender?"). Do NOT proceed with symptom questions until you have this basic information.

Once you have age and gender:
1. CROSS-QUERY MODE: Ask ONLY ONE (1) most important clarifying question to help narrow down the diagnosis.
2. DO NOT provide treatments or final answers yet.
3. Keep it brief and professional.
4. Avoid asking questions that are not relevant to Ayurvedic diagnosis.
5. Ground your question strictly in the provided sources and conversation history.
6. Gather information about lifestyle, diet, medical history, and habits as this is crucial for Ayurvedic diagnosis.

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

CONVERSATION HISTORY:
{conversation_history}

RETRIEVED SOURCES:
{context}

CURRENT USER INPUT:
{question}

INSTRUCTIONS:
{instruction}
7. Groundedness: Remain strictly inside the provided sources.
8. Ask the questions only if it make sense, for example for a male patient asking about menstruation, it would not make sense

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
