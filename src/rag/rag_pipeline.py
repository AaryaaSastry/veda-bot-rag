import config
from rag.retriever import Retriever
from rag.generator import Generator
from sentence_transformers import CrossEncoder
from rag.query_rewriter import rewrite_query


class RAGPipeline:

    def __init__(self, vector_db_path, api_key):
        self.retriever = Retriever(vector_db_path)
        self.generator = Generator(api_key)
        self.reranker = CrossEncoder(config.RERANKER_MODEL)

    def run(self, question, memory, source_filter=None):

        conversation_history = memory.get_formatted_history()
        user_turns = memory.user_turn_count

        rewritten_query = rewrite_query(
            generator=self.generator,
            conversation_history=conversation_history,
            current_question=question
        )

        # print(f"Retrieving initial {config.K_RETRIEVAL} chunks...")
        retrieved_chunks = self.retriever.retrieve(rewritten_query, k=config.K_RETRIEVAL, source_filter=source_filter)

        if not retrieved_chunks:
            return ["No relevant information found."]

        # print(f"Reranking to top {config.K_RERANK} chunks...")
        # Prepare pairs for cross-encoder
        pairs = [[rewritten_query, chunk['text']] for chunk in retrieved_chunks]
        scores = self.reranker.predict(pairs)
        
        # Add scores and sort
        for i, chunk in enumerate(retrieved_chunks):
            chunk['rerank_score'] = scores[i]
            
        reranked_chunks = sorted(
            retrieved_chunks, 
            key=lambda x: x['rerank_score'], 
            reverse=True
        )[:config.K_RERANK]

        # LOGIC FLOW
        if memory.waiting_remedies_consent:
            if any(word in question.lower() for word in ["yes", "yeah", "ok", "sure", "please", "remedies"]):
                memory.waiting_remedies_consent = False
                memory.mark_complete()
                
                # Fetch remedies specifically for the diagnosis from the knowledge base
                diag_name = "the condition"
                if memory.last_diagnosis and "DIAGNOSIS:" in memory.last_diagnosis:
                    diag_name = memory.last_diagnosis.split("DIAGNOSIS:")[1].split("\n")[0].strip()

                remedy_query = f"Ayurvedic remedies, treatments, foods, habits (do's and don'ts) for {diag_name}"
                # print(f"Searching for remedies for {diag_name}...")
                remedy_chunks = self.retriever.retrieve(remedy_query, k=config.K_RERANK, source_filter=source_filter)
                
                return self.generator.generate(remedy_query, remedy_chunks, conversation_history, mode="remedies")
            else:
                memory.waiting_remedies_consent = False
                memory.mark_complete()
                def farewell_gen():
                    yield "I understand. Please let me know if you need anything else. Goodbye!"
                return farewell_gen()

        if memory.diagnosis_complete:
            # Diagnosis is already done, just chat normally
            return self.generator.generate(question, reranked_chunks, conversation_history, mode="final")

        if user_turns < 4:
            # Phase 1: Gathering (5 questions)
            # print(f"Gathering mode: Question {user_turns + 1}/5")
            return self.generator.generate(question, reranked_chunks, conversation_history, mode="gathering")
        
        elif user_turns == 4:
            # After 5 questions, attempt diagnosis
            # print("Turn 5 reached. Generating initial diagnosis...")
            diagnosis_report = self.generator.generate_diagnosis(conversation_history, reranked_chunks)
            print(f"\n--- DIAGNOSIS REPORT ---\n{diagnosis_report}\n------------------------")
            
            print("Verifying diagnosis...")
            is_valid = self.generator.verify_diagnosis(diagnosis_report, conversation_history)
            
            if is_valid:
                print("Verification SUCCESS. Generating final answer...")
                memory.last_diagnosis = diagnosis_report
                memory.waiting_remedies_consent = True
                return self.generator.generate(question, reranked_chunks, conversation_history, mode="diagnosis")
            else:
                print("Verification FAILED. Need 5 more questions.")
                # We return a message indicating we need more info
                return self.generator.generate(question, reranked_chunks, conversation_history, mode="gathering")

        elif 4 < user_turns < 9:
            # Phase 2: Extra gathering (5 more questions)
            print(f"Extra Gathering mode: Question {user_turns - 4}/5")
            return self.generator.generate(question, reranked_chunks, conversation_history, mode="gathering")
        
        else:
            # Final Diagnosis after turn 20
            print("Final diagnosis mode reached.")
            diagnosis_report = self.generator.generate_diagnosis(conversation_history, reranked_chunks)
            memory.last_diagnosis = diagnosis_report
            memory.waiting_remedies_consent = True
            return self.generator.generate(question, reranked_chunks, conversation_history, mode="diagnosis")
