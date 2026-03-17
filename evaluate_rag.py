
import os
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import torch
import warnings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from ragas.llms import LangchainLLMWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings

# Import local modules
import load_model
try:
    import query
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import query

# Suppress warnings
warnings.filterwarnings("ignore")

from langchain_core.language_models.llms import LLM
from typing import Any, List, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

class MistralInstructLLM(LLM):
    pipeline: Any = None
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # Wrap prompt in [INST] tags for better instruction following
        formatted_prompt = f"[INST] {prompt} [/INST]"
            
        # Generate
        # We assume pipeline is valid
        outputs = self.pipeline(
            formatted_prompt, 
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            return_full_text=False
        )
        response = outputs[0]['generated_text']
        return response.strip()

    @property
    def _llm_type(self) -> str:
        return "mistral_instruct"

def setup_ragas_llm():
    """Configures the local Mistral model for RAGAS."""
    print("Setting up local LLM for RAGAS...")
    
    # Create text-generation pipeline
    pipe = pipeline(
        "text-generation",
        model=load_model.model,
        tokenizer=load_model.tokenizer,
        max_new_tokens=512,
        pad_token_id=load_model.tokenizer.pad_token_id
    )
    
    # Use our custom wrapper
    # Ragas v0.1+ wraps this in LangchainLLMWrapper automatically if passed as such?
    # Actually setup_ragas_llm expects to return a Ragas LLM or Langchain LLM.
    # If we return a Langchain LLM, Ragas will wrap it or use it.
    # But strictly speaking we should wrap it with LangchainLLMWrapper for Ragas config.
    
    mistral_llm = MistralInstructLLM(pipeline=pipe)
    return LangchainLLMWrapper(langchain_llm=mistral_llm)

def get_test_data():
    """
    Creates a small test dataset for evaluation.
    Ideally this should be loaded from a file or larger dataset.
    """
    return [
        {
            "question": "What is the C4C program?",
            "ground_truth": "The C4C (California for All College Corps) program provides up to $10,000 for college students who complete community service work."
        },
        {
            "question": "Who is eligible for the California Dream Act?",
            "ground_truth": "Undocumented students, DACA recipients, U Visa holders, and TPS status holders who meet AB 540 requirements are eligible for the California Dream Act."
        },
        {
            "question": "What financial aid is available for undocumented students?",
            "ground_truth": "Undocumented students in California can apply for the California Dream Act (CADAA), Cal Grants, Chafee Grant, Middle Class Scholarship, and institutional scholarships."
        }
    ]

def evaluate_pipeline():
    print("Starting RAG Evaluation...")
    
    # 1. Setup Wrapper LLM/Embeddings
    # Ragas needs an LLM for 'faithfulness' and 'answer_relevancy' (as critic)
    ragas_llm = setup_ragas_llm()
    
    # Setup embeddings for metrics that need it (answer_relevancy)
    # Using a lightweight local embedding model instead of OpenAI
    print("Loading embedding model for metrics...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 2. Collect Data
    data_samples = {
        'question': [],
        'answer': [],
        'contexts': [],
        'ground_truth': []
    }
    
    test_data = get_test_data()
    
    print(f"Running generation for {len(test_data)} test questions...")
    for item in test_data:
        q = item["question"]
        gt = item["ground_truth"]
        
        print(f"Processing: {q}")
        # Call our RAG pipeline
        # Note: query.generate_answer now returns (answer, sources, contexts)
        ans, sources, contexts = query.generate_answer(q, return_context=True)
        
        # Clean answer (remove Source part if present for cleaner eval, though Ragas handles it usually)
        # Our model output might contain "Sources: ...", Ragas usually just wants the text answer.
        # We can pass it as is, or strip sources.
        if "Sources:" in ans:
             ans_text = ans.split("Sources:")[0].strip()
        else:
             ans_text = ans.strip()

        data_samples['question'].append(q)
        data_samples['answer'].append(ans_text)
        data_samples['contexts'].append(contexts)  # Ragas expects list of strings
        data_samples['ground_truth'].append(gt)

    # 3. Create Dataset
    dataset = Dataset.from_dict(data_samples)
    
    # 4. Run Evaluation
    # Note: 'context_precision' and 'context_recall' generally don't utilize the LLM as judge, 
    # but 'faithfulness' and 'answer_relevancy' do.
    
    print("Calculating metrics... (this may take time on local GPU)")
    
    # Configure metrics with our local LLM and Embeddings
    # We need to manually inject them into the metrics if not using default OpenAI
    
    # Update metrics with local LLM/Embeddings
    # Ragas v0.1+ pattern
    
    faithfulness.llm = ragas_llm
    answer_relevancy.llm = ragas_llm
    answer_relevancy.embeddings = embeddings
    context_precision.llm = ragas_llm
    context_recall.llm = ragas_llm
    
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
    
    results = evaluate(
        dataset,
        metrics=metrics,
        llm=ragas_llm, # Fallback/Global
        embeddings=embeddings
    )
    
    print("\n=== Evaluation Results ===")
    print(results)
    
    # Save results
    results_df = results.to_pandas()
    results_df.to_csv("rag_evaluation_results.csv", index=False)
    print("Results saved to rag_evaluation_results.csv")

if __name__ == "__main__":
    evaluate_pipeline()


# import os
# import ragas

# __import__("pysqlite3")
# import sys
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# import warnings
# warnings.filterwarnings("ignore")

# from datasets import Dataset
# from ragas import evaluate
# from ragas.metrics import (
#     faithfulness,
#     answer_relevancy,
#     context_precision,
#     context_recall,
# )

# from transformers import pipeline
# from ragas.llms import LangchainLLMWrapper
# from langchain_community.embeddings import HuggingFaceEmbeddings

# import load_model
# import query

# from langchain_core.language_models.llms import LLM
# from typing import Any, List, Optional
# from langchain_core.callbacks.manager import CallbackManagerForLLMRun

# # =========================================================
# # Deterministic LLM wrapper for RAGAS
# # =========================================================
# class MistralInstructLLM(LLM):
#     pipeline: Any = None

#     def _call(
#         self,
#         prompt: str,
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[CallbackManagerForLLMRun] = None,
#         **kwargs: Any,
#     ) -> str:
#         formatted_prompt = f"[INST] {prompt} [/INST]"
#         outputs = self.pipeline(
#             formatted_prompt,
#             max_new_tokens=256,
#             do_sample=False,     # IMPORTANT: deterministic
#             temperature=0.0,     # IMPORTANT: no randomness
#             return_full_text=False,
#         )
#         return outputs[0]["generated_text"].strip()

#     @property
#     def _llm_type(self) -> str:
#         return "mistral_instruct"


# def setup_ragas_llm():
#     pipe = pipeline(
#         "text-generation",
#         model=load_model.model,
#         tokenizer=load_model.tokenizer,
#         pad_token_id=load_model.tokenizer.pad_token_id,
#     )
#     return LangchainLLMWrapper(
#         langchain_llm=MistralInstructLLM(pipeline=pipe)
#     )

# # =========================================================
# # Evaluation Questions (25)
# # =========================================================
# def get_test_data():
#     return [
#         {
#             "question": "What is the California Dream Act and who is eligible to apply?",
#             "ground_truth": "The California Dream Act allows eligible undocumented and nonresident students to apply for state financial aid if they meet residency and academic requirements."
#         },
#         {
#             "question": "What types of financial aid are available to undocumented students in California?",
#             "ground_truth": "Undocumented students in California may qualify for state financial aid such as Cal Grants, the California Dream Act Application, scholarships, and institutional aid."
#         },
#         {
#             "question": "What is the California for All College Corps program?",
#             "ground_truth": "The California for All College Corps program provides paid community service opportunities for college students."
#         },
#         {
#             "question": "What is a Cal Grant?",
#             "ground_truth": "A Cal Grant is a California state financial aid program that helps eligible students pay for college based on financial need and academic criteria."
#         },
#         {
#             "question": "What does the California Student Aid Commission do?",
#             "ground_truth": "The California Student Aid Commission administers state financial aid programs such as Cal Grants and the California Dream Act."
#         },
#         {
#             "question": "What is the FAFSA used for?",
#             "ground_truth": "The FAFSA is used to apply for federal financial aid, including grants, loans, and work-study programs."
#         },
#         {
#             "question": "What is federal work-study?",
#             "ground_truth": "Federal work-study provides part-time employment opportunities to help students pay for college expenses."
#         },
#         {
#             "question": "What is the difference between grants and loans?",
#             "ground_truth": "Grants do not need to be repaid, while loans must be repaid with interest."
#         },
#         {
#             "question": "Who is considered a first-generation college student?",
#             "ground_truth": "A first-generation college student is someone whose parents or guardians did not complete a four-year college degree."
#         },
#         {
#             "question": "What resources help first-generation college students?",
#             "ground_truth": "Resources include mentoring programs, academic advising, financial aid guidance, and campus support services."
#         },
#         {
#             "question": "How can students research colleges effectively?",
#             "ground_truth": "Students can research colleges by reviewing academic programs, costs, admission requirements, and campus support services."
#         },
#         {
#             "question": "What factors should be considered when building a college list?",
#             "ground_truth": "Factors include academic fit, cost, location, campus size, and financial aid availability."
#         },
#         {
#             "question": "What are alternatives to attending a four-year college?",
#             "ground_truth": "Alternatives include community college, vocational training, apprenticeships, or entering the workforce."
#         },
#         # {
#         #     "question": "What is in-state tuition in California?",
#         #     "ground_truth": "In-state tuition is a reduced tuition rate for students who meet California residency or eligibility requirements."
#         # },
#         # {
#         #     "question": "What is institutional financial aid?",
#         #     "ground_truth": "Institutional financial aid is provided by colleges to help students cover educational costs."
#         # },
#         {
#             "question": "What are Pell Grants?",
#             "ground_truth": "Pell Grants are federal grants awarded to undergraduate students with significant financial need."
#         },
#         {
#             "question": "What should students review in a financial aid offer?",
#             "ground_truth": "Students should review grants, scholarships, loans, work-study options, and total cost of attendance."
#         },
#         {
#             "question": "What are private student loans?",
#             "ground_truth": "Private student loans are offered by private lenders and often have different terms than federal loans."
#         },
#         {
#             "question": "What support exists for undocumented students in higher education?",
#             "ground_truth": "Support includes financial aid guidance, legal resources, campus support centers, and nonprofit organizations."
#         },
#         {
#             "question": "What is the role of a high school counselor?",
#             "ground_truth": "High school counselors assist with course planning, college applications, and financial aid guidance."
#         },
#         {
#             "question": "What is the BigFuture platform used for?",
#             "ground_truth": "BigFuture helps students explore colleges, careers, scholarships, and financial aid."
#         },
#         {
#             "question": "What is the Higher Ed Immigration Portal?",
#             "ground_truth": "It provides state-specific immigration and higher education resources."
#         },
#         {
#             "question": "What repayment options exist for student loans?",
#             "ground_truth": "Options include standard repayment, income-driven repayment, and loan forgiveness programs."
#         },
#         {
#             "question": "What are common scholarship application mistakes?",
#             "ground_truth": "Common mistakes include missing deadlines and submitting incomplete applications."
#         },
#         # {
#         #     "question": "What is federal financial aid?",
#         #     "ground_truth": "Federal financial aid includes grants, loans, and work-study programs funded by the U.S. government."
#         # },
#     ]

# # =========================================================
# # Evaluation Pipeline
# # =========================================================
# def evaluate_pipeline():
#     print("Starting RAGAS evaluation...")

#     ragas_llm = setup_ragas_llm()
#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )

#     data = {
#         "question": [],
#         "answer": [],
#         "contexts": [],
#         "ground_truth": [],
#     }

#     skipped = 0
#     errors = []

#     for item in get_test_data()[:15]:
#         q = item["question"]
#         gt = item["ground_truth"]

#         try:
#             answer, _, contexts = query.generate_answer(q, return_context=True)

#             # VALIDATION: skip empty-context samples
#             if not contexts or len(contexts) == 0:
#                 print(f"[skip] No context retrieved for: {q}")
#                 skipped += 1
#                 continue

#             # VALIDATION: ensure all contexts are non-empty strings
#             contexts = [str(c).strip() for c in contexts if c and str(c).strip()]
#             if not contexts:
#                 print(f"[skip] All contexts empty after cleaning: {q}")
#                 skipped += 1
#                 continue

#             # VALIDATION: ensure answer is non-empty
#             answer_text = answer.split("Sources:")[0].strip()
#             if not answer_text:
#                 print(f"[skip] Empty answer for: {q}")
#                 skipped += 1
#                 continue

#             # VALIDATION: ensure ground truth is non-empty
#             if not gt or not str(gt).strip():
#                 print(f"[skip] Empty ground truth for: {q}")
#                 skipped += 1
#                 continue

#             data["question"].append(q)
#             data["answer"].append(answer_text)
#             data["contexts"].append(contexts)
#             data["ground_truth"].append(gt)
#         except Exception as e:
#             print(f"[error] Processing failed for '{q}': {str(e)}")
#             errors.append((q, str(e)))
#             skipped += 1
#             continue

#     print(f"\nEvaluation samples used: {len(data['question'])}")
#     print(f"Samples skipped: {skipped}")
#     if errors:
#         print(f"Errors encountered: {len(errors)}")
#         for q, err in errors[:3]:
#             print(f"  - {q}: {err}")

#     if len(data['question']) < 3:
#         print("[ERROR] Not enough valid samples for evaluation (need at least 3)")
#         return

#     dataset = Dataset.from_dict(data)

#     # Configure metrics with LLM and embeddings BEFORE evaluation
#     faithfulness.llm = ragas_llm
#     answer_relevancy.llm = ragas_llm
#     answer_relevancy.embeddings = embeddings
#     context_precision.llm = ragas_llm
#     context_recall.llm = ragas_llm

#     print("\nRunning RAGAS evaluation (this may take a few minutes)...\n")
    
#     results = evaluate(
#         dataset,
#         metrics=[
#             faithfulness,
#             answer_relevancy,
#             context_precision,
#             context_recall,
#         ],
#         llm=ragas_llm,
#         embeddings=embeddings,
#         raise_exceptions=False,
#     )

#     print("\n=== RAGAS RESULTS ===")
#     print(results)

#     # Print detailed statistics
#     results_df = results.to_pandas()
#     print("\n=== DETAILED STATS ===")
#     for col in results_df.columns:
#         if col not in ['question', 'answer', 'contexts', 'ground_truth']:
#             valid_values = results_df[col].dropna()
#             if len(valid_values) > 0:
#                 print(f"{col}:")
#                 print(f"  Mean: {valid_values.mean():.4f}")
#                 print(f"  Std:  {valid_values.std():.4f}")
#                 print(f"  Valid: {len(valid_values)}/{len(results_df)}")
#             else:
#                 print(f"{col}: ALL NaN VALUES")

#     results_df.to_csv("rag_evaluation_results.csv", index=False)
#     print("\nSaved results to rag_evaluation_results.csv")

# # =========================================================
# # Entry point
# # =========================================================
# if __name__ == "__main__":
#     evaluate_pipeline()
