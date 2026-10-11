# Energy Operations Document RAG Assistant

**Context:** Greenko operates complex renewable-energy assets and enterprise workflows where AI can support operations, maintenance, safety, analytics, and decision support.

## Problem Statement

Build a document intelligence assistant for renewable-energy operations, safety, and maintenance SOPs.

## Expected Outcome

A working RAG-based assistant that answers operational questions from provided documents with citations, retrieval evidence, and hallucination checks.

## Skills Tested

RAG, LLM prompting, chunking, embeddings, retrieval, evaluation, Python API, frontend prototype

## Candidate Submission Requirements

1. Build the Python backend or notebook pipeline.
2. Build a simple React or HTML/CSS frontend to demonstrate the workflow.
3. Use the bundled dummy dataset as the minimum runnable dataset.
4. Add clear documentation explaining design choices, assumptions, model logic, limitations, and evaluation results.
5. Deploy the completed app to Railway.

## Dataset Note

This package includes a small self-contained dummy dataset so the assignment can run without internet access. Candidates may optionally extend the dataset using public sources such as Hugging Face datasets, Kaggle, Open Power System Data, NREL examples, or other open energy/NLP datasets, but the core submission must run using the bundled data.

## Suggested Deliverables

- Working Python implementation
- API endpoint or executable script
- UI prototype
- Model / retrieval / classification output examples
- Evaluation results
- Short explanation of production-readiness gaps

# Evaluation Rubric

Score each submission out of 100 using the same Greenko AI/ML intern screening philosophy.

| Area | Weight |
|---|---:|
| Problem understanding and energy-domain framing | 10 |
| Python implementation quality | 12 |
| AI/ML or GenAI method selection | 14 |
| Data preprocessing and feature engineering | 10 |
| RAG / retrieval / agentic workflow quality, where applicable | 12 |
| Evaluation metrics and validation approach | 12 |
| API/backend design | 8 |
| Frontend usability and explanation clarity | 8 |
| Documentation and reproducibility | 8 |
| Deployment readiness, including Railway deployment step | 6 |

Strong submissions should clearly explain assumptions, limitations, failure cases, and next steps for productionization.

---

# Implementation Notes — Candidate Submission

## 1. Solution Overview

The Energy Operations Document RAG Assistant is a FastAPI-based application that answers renewable-energy operations questions using the supplied Markdown documents. It provides a simple HTML frontend and returns answers alongside source citations and retrieval evidence.

The current implementation combines TF-IDF-based document retrieval with OpenAI answer generation.

## 2. Architecture and Model Logic

### Document processing

- Input documents: Markdown files in the `data/` directory.
- Chunk size: 100 words.
- Chunk overlap: 20 words.
- Chunk identifiers and source filenames are retained for traceability.

### Retrieval

- Method: TF-IDF vectorization with unigrams and bigrams.
- Text normalization: Lowercasing and English stop-word removal.
- Ranking: Cosine similarity between the question and document chunks.
- Number of retrieved chunks: Up to 3.
- Current minimum similarity threshold: 0.08.

### Answer generation

When sufficient retrieval evidence is available and an OpenAI API key is configured, the backend uses the `gpt-4o-mini` model to generate an answer from the retrieved passages.

The prompt instructs the model to use the supplied evidence, avoid inventing operational facts, and cite supporting filenames.

When the OpenAI key is not configured, the backend uses a retrieved-excerpt fallback. This fallback is not an LLM-generated answer.

### API endpoints

- `GET /` — Serves the HTML frontend.
- `GET /health` — Reports application health and loaded chunk count.
- `GET /api/demo` — Returns application and retrieval information.
- `POST /api/run` — Retrieves evidence and returns an answer, citations, retrieval evidence, and abstention status.

## 3. Configuration and Local Setup

Requirements: Python 3.12 or another compatible Python version, the project dependencies, and an OpenAI API key for generated answers.

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root:

```dotenv
OPENAI_API_KEY=your_actual_api_key
```

Do not commit `.env` or expose the API key in source code, screenshots, or public repositories.

Start the application:

```bash
python -m uvicorn backend.app:app --reload
```

Open `http://127.0.0.1:8000` to use the frontend and `http://127.0.0.1:8000/docs` to inspect the API.

## 4. Evaluation Results

The bundled `data/questions.csv` contains two questions with expected source documents.

| Test question | Expected source | Observed retrieval result |
|---|---|---|
| What should be checked for inverter derating? | `solar_inverter_sop.md` | Correct source retrieved; similarity score 0.2867 |
| How should gearbox vibration be investigated? | `wind_turbine_maintenance.md` | Correct source retrieved; similarity score 0.3087 |

Observed expected-source retrieval success: 2 out of 2 questions (100%) on this small supplied evaluation set.

An additional manual test asked about Greenko's employee maternity leave policy. The system returned an insufficient-evidence message, with abstention enabled and no citations or retrieved passages.

These results are preliminary. They measure the observed source retrieval outcomes for two supplied questions and one unsupported-question test; they do not establish general answer accuracy or prove that all hallucinations are prevented.

## 5. Assumptions and Limitations

- TF-IDF retrieval is lexical and may miss relevant passages when question wording differs from the documents.
- The chunk size, overlap, top-k value, and similarity threshold are initial heuristic choices.
- A similarity threshold alone cannot guarantee that retrieved passages are relevant.
- The answer-generation prompt encourages evidence-grounded answers but is not an independent factuality verification system.
- Returned source metadata identifies the retrieved passages; generated claims still require validation against those passages.
- Evaluation coverage is limited by the small bundled dataset.
- The application is a prototype, not a validated system for autonomous safety-critical operational decisions.
- Production use would require broader evaluation, stronger factuality checks, access control, monitoring, and appropriate human review.

## 6. Deployment Status

The application has been tested locally. Railway deployment and the deployed URL must be added here after deployment has been completed and verified.

- Railway deployment URL: Pending verification
- Deployed health endpoint: Pending verification
- Deployed question-answering test: Pending verification
