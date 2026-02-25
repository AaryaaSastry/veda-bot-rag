def rewrite_query(generator, conversation_history, current_question):
    prompt = f"""
Given the following conversation history and a new sub-question, your task is to rewrite the sub-question into a standalone, descriptive search query that captures the user's information need.

CONVERSATION HISTORY:
{conversation_history}

NEW QUESTION:
{current_question}

INSTRUCTIONS:
1. If the new question is shorthand (like "?", "tell me more", or "what else?"), expand it based on what was just discussed.
2. If it is already a complete question, refine it for better search results.
3. If it is a greeting or social remark, just return it as is.
4. ONLY output the rewritten search query. No preamble.

Rewritten Query:
"""

    rewritten = generator.generate_text(prompt)
    return rewritten.strip()