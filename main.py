#!/usr/bin/env python3
from typing import List, Optional, Any
import os
import logging
from fastapi import FastAPI, Request, Header, HTTPException
from pydantic import BaseModel
import requests
import numpy as np
import uvicorn
import re
import json

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rerank-bridge")

# Configuration via environment variables
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")
BRIDGE_API_KEY = os.environ.get("RERANK_BRIDGE_API_KEY", "")
RERANK_MODE = os.environ.get("RERANK_MODE", "embeddings")  # embeddings | generate
PORT = int(os.environ.get("RERANK_PORT", "8000"))
TIMEOUT = int(os.environ.get("RERANK_TIMEOUT", "30"))

app = FastAPI(title="OpenWebUI ↔ Ollama Rerank Bridge")

class RerankRequest(BaseModel):
    model: str
    query: str
    documents: List[str]
    top_n: Optional[int] = None

def _headers_for_ollama():
    h = {"Content-Type": "application/json"}
    if OLLAMA_API_KEY:
        h["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
    log.debug(f"Ollama headers: {h}")
    return h

def _cosine(a: List[float], b: List[float]) -> float:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def _normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return scores
    s_min = min(scores)
    s_max = max(scores)
    if s_max - s_min == 0:
        # If all equal, produce 1.0 for the max and scaled otherwise
        return [1.0 if s == s_max else 0.0 for s in scores]
    return [(s - s_min) / (s_max - s_min) for s in scores]

def _extract_embeddings_from_ollama_response(data: Any):
    # Try common response shapes
    log.debug(f"Extracting embeddings from response: {type(data)}")
    if isinstance(data, dict):
        log.debug(f"Response is dict with keys: {data.keys()}")
        if "embeddings" in data:
            log.debug(f"Found 'embeddings' key, returning {len(data['embeddings'])} embeddings")
            return data["embeddings"]
        if "data" in data and isinstance(data["data"], list):
            out = []
            for item in data["data"]:
                if isinstance(item, dict) and "embedding" in item:
                    out.append(item["embedding"])
            if out:
                log.debug(f"Found 'data' key with {len(out)} embeddings")
                return out
    if isinstance(data, list):
        log.debug(f"Response is list with {len(data)} items")
        return data
    log.error(f"Unexpected embeddings response shape: {data}")
    raise ValueError("Unexpected embeddings response shape from Ollama")

@app.post("/v1/rerank")
async def rerank(request: Request, body: RerankRequest, authorization: Optional[str] = Header(None)):
    # Optional API key enforcement
    if BRIDGE_API_KEY:
        if not authorization or not authorization.startswith("Bearer ") or authorization.split(" ",1)[1] != BRIDGE_API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")

    model = body.model
    query = body.query
    docs = body.documents
    top_n = body.top_n or len(docs)

    log.info(f"Received rerank request: model={model}, query='{query}', docs={len(docs)}, top_n={top_n}")

    if RERANK_MODE == "embeddings":
        # Embeddings mode: request embeddings for [query] + docs and compute cosines
        texts = [query] + docs
        try:
            # Use /api/embed with batch processing - send all texts in one request
            log.info(f"Sending batch embedding request for {len(texts)} texts to Ollama: {OLLAMA_BASE_URL}/api/embed")
            payload = {"model": model, "input": texts}
            log.debug(f"Payload: {payload}")
            
            try:
                r = requests.post(f"{OLLAMA_BASE_URL}/api/embed", json=payload, headers=_headers_for_ollama(), timeout=TIMEOUT)
                log.info(f"Response status: {r.status_code}")
            except Exception as req_err:
                log.error(f"Request failed: {req_err}")
                raise
            
            if r.status_code != 200:
                log.error(f"Ollama returned error: {r.text}")
            r.raise_for_status()
            data = r.json()
            
            # Extract embeddings from batch response
            # Ollama returns {"embeddings": [[...], [...], ...]} for batch input
            if "embeddings" in data and isinstance(data["embeddings"], list):
                embeddings = data["embeddings"]
                if len(embeddings) != len(texts):
                    log.error(f"Expected {len(texts)} embeddings, got {len(embeddings)}")
                    raise HTTPException(status_code=502, detail="Ollama returned wrong number of embeddings")
            else:
                log.error(f"No 'embeddings' key in response: {data}")
                raise HTTPException(status_code=502, detail="Ollama did not return embeddings")
            
            if len(embeddings) < len(texts):
                log.error(f"Expected {len(texts)} embeddings, got {len(embeddings)}")
                raise HTTPException(status_code=502, detail="Not enough embeddings returned from Ollama")
            
            q_emb = embeddings[0]
            doc_embs = embeddings[1:1+len(docs)]
            raw_scores = [ _cosine(q_emb, d) for d in doc_embs ]
            norm_scores = _normalize_scores(raw_scores)
            results = [{"index": i, "relevance_score": float(norm_scores[i])} for i in range(len(norm_scores))]
            # sort descending by score and slice top_n
            results_sorted = sorted(results, key=lambda x: x["relevance_score"], reverse=True)[:top_n]
            log.info(f"Rerank successful. Returning {len(results_sorted)} results.")
            return {"results": results_sorted}
        except HTTPException:
            raise
        except Exception as e:
            log.exception("Error calling Ollama embeddings")
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

    elif RERANK_MODE == "generate":
        # Generate mode: call Ollama generate API with a prompt to return a numeric score for each doc.
        results = []
        for i, doc in enumerate(docs):
            prompt = (
                f"Rate the relevance of the document to the query on a scale 0.0-1.0.\n\n"
                f"Query: {query}\n\nDocument: {doc}\n\n"
                f"Respond with a single JSON object exactly: {{\"index\": {i}, \"relevance_score\": <score>}}"
            )
            payload = {"model": model, "prompt": prompt, "stream": False}
            try:
                r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, headers=_headers_for_ollama(), timeout=TIMEOUT)
                r.raise_for_status()
                data = r.json()
                out_text = ""
                if isinstance(data, dict):
                    if "text" in data:
                        out_text = data["text"]
                    elif "choices" in data and isinstance(data["choices"], list) and "text" in data["choices"][0]:
                        out_text = data["choices"][0]["text"]
                    elif "results" in data and isinstance(data["results"], list) and "content" in data["results"][0]:
                        out_text = data["results"][0]["content"]
                if not out_text:
                    out_text = json.dumps(data)
                m = re.search(r"([0-1](?:\.\d+)?|\d\.\d+)", out_text)
                score = float(m.group(0)) if m else 0.0
                results.append({"index": i, "relevance_score": score})
            except Exception as e:
                log.exception("Error calling Ollama generate")
                results.append({"index": i, "relevance_score": 0.0})
        results_sorted = sorted(results, key=lambda x: x["relevance_score"], reverse=True)[:top_n]
        return {"results": results_sorted}
    else:
        raise HTTPException(status_code=500, detail=f"Unknown RERANK_MODE: {RERANK_MODE}")

@app.get("/health")
def health():
    return {"status":"ok", "mode": RERANK_MODE}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, log_level="info")