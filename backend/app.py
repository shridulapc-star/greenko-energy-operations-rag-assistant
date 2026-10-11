
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

app = FastAPI(title="Energy Operations Document RAG Assistant")

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
FRONTEND = BASE / "frontend"


CHUNK_SIZE = 100
CHUNK_OVERLAP = 20
TOP_K = 3
MIN_SIMILARITY = 0.08


class Query(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


def load_chunks():
    chunks = []

    for path in sorted(DATA.glob("*.md")):
        words = path.read_text(encoding="utf-8").split()
        step = CHUNK_SIZE - CHUNK_OVERLAP

        for start in range(0, len(words), step):
            text = " ".join(words[start:start + CHUNK_SIZE])
            if text:
                chunks.append({
                    "source": path.name,
                    "chunk_id": f"{path.stem}-{start // step + 1}",
                    "text": text,
                })

    return chunks


CHUNKS = load_chunks()
VECTORIZER = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
)
MATRIX = (
    VECTORIZER.fit_transform([c["text"] for c in CHUNKS])
    if CHUNKS else None
)


def retrieve(question: str):
    if MATRIX is None:
        return []

    vector = VECTORIZER.transform([question])
    scores = cosine_similarity(vector, MATRIX).ravel()
    indices = scores.argsort()[::-1][:TOP_K]

    return [
        {**CHUNKS[i], "score": round(float(scores[i]), 4)}
        for i in indices if scores[i] > 0
    ]


@app.get("/")
def home():
    return FileResponse(FRONTEND / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "chunks_loaded": len(CHUNKS)}


@app.get("/api/demo")
def demo():
    return {
        "case_study": "Energy Operations Document RAG Assistant",
        "status": "running",
        "chunks_loaded": len(CHUNKS),
        "retrieval_method": "TF-IDF and cosine similarity",
    }



@app.post("/api/run")
def run_case(query: Query):
    question = query.text.strip()
    evidence = retrieve(question)

    # Preserve the existing evidence threshold and abstention behavior.
    if not evidence or evidence[0]["score"] < MIN_SIMILARITY:
        return {
            "input": question,
            "result": (
                "Insufficient evidence in the supplied documents. "
                "Consult the relevant operational SOP or responsible team."
            ),
            "abstained": True,
            "citations": [],
            "retrieval_evidence": evidence,
            "retrieval_method": "TF-IDF and cosine similarity",
        }

    # Keep the retrieval-only baseline available if no key is configured.
    if client is None:
        answer = "\n\n".join(
            f"[{item['source']}] {item['text']}"
            for item in evidence
        )
        return {
            "input": question,
            "result": answer,
            "abstained": False,
            "citations": [
                {
                    "source": item["source"],
                    "chunk_id": item["chunk_id"],
                }
                for item in evidence
            ],
            "retrieval_evidence": evidence,
            "retrieval_method": "TF-IDF and cosine similarity",
            "answer_generation": "Extractive fallback (OpenAI not configured)",
            "limitations": [
                "The answer contains retrieved excerpts, not a generated summary.",
                "The similarity threshold is heuristic and requires evaluation.",
            ],
        }

    context = "\n\n".join(
        f"Source: {item['source']}\n"
        f"Chunk ID: {item['chunk_id']}\n"
        f"Passage: {item['text']}"
        for item in evidence
    )

    prompt = f"""
You are an assistant for renewable-energy operations.

Answer the question using only the supplied document passages.

Rules:
- Do not use outside knowledge or invent operational facts.
- If the passages do not contain enough information to answer,
  explicitly say that the evidence is insufficient.
- Give a concise, practical answer.
- Cite supporting filenames in square brackets, such as
  [solar_inverter_sop.md].
- Do not invent citations or claim that a source supports something
  it does not say.
- For safety-critical instructions, preserve the meaning of the source.

DOCUMENT PASSAGES:
{context}

USER QUESTION:
{question}
"""

    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            instructions=(
                "Be evidence-grounded and cautious. "
                "Never invent operational procedures or source references."
            ),
            input=prompt,
        )
        answer = response.output_text.strip()

        if not answer:
            raise HTTPException(
                status_code=502,
                detail="The answer-generation service returned no text.",
            )

    except HTTPException:
        raise
    except Exception as exc:
        # Do not expose credentials or raw provider error details.
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI request failed ({type(exc).__name__}).",
        ) from exc

    citations = [
        {
            "source": item["source"],
            "chunk_id": item["chunk_id"],
        }
        for item in evidence
    ]

    return {
        "input": question,
        "result": answer,
        "abstained": False,
        "citations": citations,
        "retrieval_evidence": evidence,
        "retrieval_method": "TF-IDF and cosine similarity",
        "answer_generation": "OpenAI gpt-4o-mini",
        "limitations": [
            "Retrieval uses lexical TF-IDF rather than semantic embeddings.",
            "Retrieved passages and generated claims require validation.",
            "The similarity threshold is heuristic.",
        ],
    }
