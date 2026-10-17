# 🏥 Shifa Hospital Assistant

**An AI helper that answers hospital staff questions using the hospital's own policy documents, and never makes things up.**

> ⚠️ **DEMO PROJECT.** "SHIFA General Hospital" is a fictional hospital and all documents are synthetic (made up for testing). This tool is **not** for real medical use and does **not** diagnose or treat patients.

---

## 1. What is this project? (in simple words)

Every hospital has many policy documents: how to admit a patient, how much a bed costs, what to do after a needle-stick injury, and so on. Finding one answer inside many PDF and Word files takes time.

**Shifa Hospital Assistant** solves this. A staff member types a question, just like chatting on WhatsApp, and the assistant:

1. Searches all the hospital documents.
2. Finds the most useful parts.
3. Writes a short answer using **only** those parts.
4. Shows **which document and page** the answer came from.

If the answer is **not** in the documents, it says so honestly instead of guessing. In a hospital, a wrong answer is worse than no answer.

---

## 2. What can it do?

| Feature | What it means for you |
|---|---|
| 💬 **Ask in plain language** | Type a normal question. No special commands needed. |
| 🌐 **English and Hinglish** | Ask in English or in Hinglish (Hindi/Urdu written in English letters), for example: *"ICU mein admission ka process kya hai?"* The answer comes back in the same style. |
| 📚 **Shows its sources** | Every answer lists the document name, department and page, so staff can check it. |
| 🚫 **Says "I don't know"** | If the documents do not contain the answer, it refuses instead of inventing one. |
| ⚡ **Live typing effect** | The answer appears word by word, like ChatGPT, so you do not wait for the whole answer. |

---

## 3. Example

**Question:** *How much does an ICU bed cost per day and what deposit is required?*

**Answer (example):** The assistant gives the daily charge and deposit from the Room and Bed Charges document, then shows a "Sources" box with the file name and page.

**Question:** *What is the monthly salary of a staff nurse?*

**Answer:** *"The answer could not be found in the available documents."* (Salary is not in any document, so it does not guess.)

---

## 4. How does it work? (no technical knowledge needed)

Think of a **very fast librarian**:

1. **Reading and cutting (done once):** The assistant reads all 16 documents and cuts them into small pieces, like index cards.
2. **Understanding meaning:** Each card is converted into a list of numbers that captures its *meaning*. Cards about similar topics get similar numbers. A special model (an "embedding model") does this, and it understands both English and Hinglish.
3. **Filing:** All cards are stored in a fast search system called an *index*.
4. **When you ask a question:** Your question is also turned into numbers, and the assistant picks the cards whose numbers are closest, meaning the most related ones.
5. **Writing the answer:** Those cards are handed to an AI language model (the "writer") with a strict instruction: *"Answer only from these cards. If it's not here, say you don't know."*
6. **Showing the result:** You see the answer, and the sources it used.

This method is called **RAG** (Retrieval-Augmented Generation). *Retrieval* = finding the right pieces. *Generation* = writing the answer from them.

```
 Your question
      │
      ▼
 Find the most related document pieces
      │
      ▼
 AI writes an answer using ONLY those pieces
      │
      ▼
 Answer + sources shown on screen
 (or "not found" if the documents don't have it)
```

---

## 5. What documents does it know?

16 synthetic documents across 6 departments:

- Emergency
- Admissions
- Patient Safety
- Pharmacy
- Nursing
- Billing & Finance

They are in PDF, Word, text and CSV (spreadsheet) formats and are stored in the `documents/` folder.

---

## 6. Technology used

| Tool | What it does (simple) |
|---|---|
| **Python** | The programming language everything is written in |
| **Streamlit** | Builds the web page (chat screen) |
| **sentence-transformers** | Turns text into numbers that capture meaning (model: `paraphrase-multilingual-MiniLM-L12-v2`) |
| **FAISS** | Searches the number-cards very quickly |
| **Groq** | Runs the AI language model that writes the answer |

---

## 7. What is inside this project?

| File / folder | Purpose |
|---|---|
| `app.py` | The chat screen people use |
| `document_loader.py` | Reads the PDF, Word, text and CSV files |
| `chunker.py` | Cuts documents into small pieces |
| `embedder.py` | Turns pieces and questions into numbers |
| `build_index.py` | Builds the search index (run once, or when documents change) |
| `retriever.py` | Finds the best matching pieces for a question |
| `rag_pipeline.py` | Connects everything and asks the AI to write the answer |
| `eval_retrieval.py` | A test script to check how well searching works |
| `evaluation_questions.csv` | Sample test questions with expected answers |
| `documents/` | The hospital documents |
| `knowledge_base/` | The saved search index (created by `build_index.py`) |

---

## 8. How to run it on your computer

You need: **Python 3.10 or newer** and a **free Groq account** for an API key.

**Step 1: Download the project**
```bash
git clone https://github.com/saadi223/shifa-knowledge-assistant
cd shifa-knowledge-assistant
```

**Step 2: Install the required tools**
```bash
pip install -r requirements.txt
```

**Step 3: Add your Groq key**
Create a file at `.streamlit/secrets.toml` and write:
```toml
GROQ_API_KEY = "paste-your-key-here"
```
Never share this key or upload this file to GitHub.

**Step 4: Build the search index (one time)**
```bash
python build_index.py
```
The first time, a language model is downloaded, so it can take a few minutes. You must run this step again whenever documents change, or when the embedding model changes. If the saved index was made with a different model, the app will stop and tell you to rebuild.

**Step 5: Start the app**
```bash
streamlit run app.py
```
Your browser opens the chat screen. Try one of the example questions in the sidebar.

---

## 9. How do we check it works well?

Run the test script:
```bash
python eval_retrieval.py --csv evaluation_questions.csv
```
For every test question it shows the best document found and a score. This helps us choose the **minimum score** below which the assistant should say "not found" (`MIN_SCORE` in `retriever.py`). Add `--llm` to also see whether the full assistant answered or refused (needs the Groq key).

---

## 10. Limitations (being honest)

- It only knows what is in the 16 documents. It has no other knowledge on purpose.
- Hinglish works with English letters only. Hindi/Urdu in Devanagari or Urdu script is not supported.
- Hinglish search quality depends on the embedding model and should be tested with `eval_retrieval.py`.
- An AI can still make mistakes, so staff should always check the shown sources.
- It is a demo with fictional data, not a medical device.

---

## 11. Small dictionary

- **RAG:** Find the right information first, then let AI write the answer from it.
- **Embedding:** Turning text into numbers that represent its meaning.
- **Index (FAISS):** A fast "search catalogue" of all those numbers.
- **LLM:** A large AI language model that can write text. Here it runs on Groq.
- **Hinglish:** Hindi/Urdu written using English letters.
- **Streaming:** Showing the answer word by word as it is written.

---

## 12. Disclaimer

This project uses fictional data for learning and demonstration. It is a knowledge/policy assistant, not a medical diagnosis system. Do not use it for real patient care.
