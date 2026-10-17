"""
eval_retrieval.py
Run this AFTER rebuilding the index. It prints, for every test question:
  - the top retrieved chunk and its similarity score
  - whether the top chunk is from the expected document
  - (with --llm) whether the full pipeline ANSWERED or REFUSED
Use the scores to choose MIN_SCORE in retriever.py. Nothing here is pre-computed.

Usage:
  python eval_retrieval.py                  # retrieval only, no Groq key needed
  python eval_retrieval.py --llm            # also call Groq (needs GROQ_API_KEY)
  python eval_retrieval.py --csv evaluation_questions.csv   # your existing 24 questions too
"""

import argparse
import csv

from retriever import load_knowledge_base, retrieve, MIN_SCORE

# Edit freely. Format: (question, language, answer_exists, expected_document)
# Expected documents come from your evaluation_questions.csv. Check them against your files.
TEST_QUESTIONS = [
    # English, answer exists
    ("How much does an ICU bed cost per day and what deposit is required?", "English", True, "BF-04_Room_and_Bed_Charges.csv"),
    ("Which two identifiers must staff use to confirm a patient's identity?", "English", True, "PS-01_Patient_Identification_Policy.pdf"),
    ("What are the OPD pharmacy timings?", "English", True, "PH-03_Pharmacy_Counter_Timings.csv"),
    # Hinglish, answer exists
    ("ICU bed ka ek din ka kharcha kitna hai aur deposit kitna dena padta hai?", "Hinglish", True, "BF-04_Room_and_Bed_Charges.csv"),
    ("Credit card se payment karne par koi extra fee lagti hai kya?", "Hinglish", True, "BF-02_Accepted_Payment_Methods.csv"),
    ("Needle-stick injury ho jaye to kya karna chahiye?", "Hinglish", True, "PS-02_Infection_Prevention_Guidelines.docx"),
    # Out of scope, must be refused
    ("How can an outpatient book an MRI scan?", "English", False, None),
    ("What is the recommended paracetamol dose for a 5-year-old child?", "English", False, None),
    ("Visitors ke liye parking fee kitni hai?", "Hinglish", False, None),
    ("Staff nurse ki monthly salary kitni hai?", "Hinglish", False, None),
]


def load_csv_questions(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            exists = r["answer_available"].strip().lower() == "yes"
            expected = r["expected_document"] if exists else None
            rows.append((r["question"], "English", exists, expected))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="also run the full pipeline (calls Groq)")
    parser.add_argument("--csv", help="extra questions file, e.g. evaluation_questions.csv")
    args = parser.parse_args()

    questions = list(TEST_QUESTIONS)
    if args.csv:
        questions += load_csv_questions(args.csv)

    index, chunks = load_knowledge_base()

    client = None
    if args.llm:
        from rag_pipeline import get_groq_client, answer_question
        client = get_groq_client()

    in_scope_scores, out_scope_scores = [], []

    for question, lang, exists, expected_doc in questions:
        # min_score=-1.0 so we always see the real top score, even if it is low
        results = retrieve(question, index, chunks, top_k=1, min_score=-1.0)
        top = results[0] if results else None

        print("=" * 90)
        print(f"[{lang}] [{'IN SCOPE' if exists else 'OUT OF SCOPE'}] {question}")
        if top is None:
            print("  No chunk returned.")
            continue

        print(f"  Top score : {top['score']:.3f}   (current MIN_SCORE = {MIN_SCORE})")
        print(f"  Top chunk : {top['filename']} (page {top['page']})")
        print(f"  Text      : {top['text'][:150].replace(chr(10), ' ')}...")
        if exists:
            print(f"  Expected doc matched: {top['filename'] == expected_doc}")
            in_scope_scores.append(top["score"])
        else:
            out_scope_scores.append(top["score"])

        if client is not None:
            result = answer_question(question, index, chunks, client)
            # Heuristic: a refusal cites no sources
            outcome = "ANSWERED" if result["cited_sources"] else "REFUSED"
            print(f"  Pipeline  : {outcome}")
            print(f"  Answer    : {result['answer'][:200].replace(chr(10), ' ')}")

    print("\n" + "=" * 90)
    print("SUMMARY (use this to choose MIN_SCORE in retriever.py)")
    if in_scope_scores:
        print(f"  In-scope questions     : lowest top score = {min(in_scope_scores):.3f}")
    if out_scope_scores:
        print(f"  Out-of-scope questions : highest top score = {max(out_scope_scores):.3f}")
    if in_scope_scores and out_scope_scores and min(in_scope_scores) <= max(out_scope_scores):
        print("  The ranges overlap: a score threshold alone cannot separate them. "
              "Keep the LLM refusal rule and choose a low threshold.")


if __name__ == "__main__":
    main()
