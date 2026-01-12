# OpenWebUI-Ollama-ReRanker-Bridge

A small, Docker-deployable bridge that implements the OpenWebUI external reranker contract and forwards rerank requests to an Ollama instance.

## API

**POST /v1/rerank**
```json
{
  "model": "<model-name>",
  "query": "<query-text>",
  "documents": ["doc0", "doc1", ...],
  "top_n": 3
}
```

**Returns:**
```json
{
  "results": [
    {"index": 0, "relevance_score": 0.92},
    {"index": 1, "relevance_score": 0.12},
    ...
  ]
}
```

**GET /health** - Health check endpoint

## Design decisions
- Ollama-only (no public APIs). Uses Ollama embeddings endpoint by default (deterministic cosine similarity).
- Lightweight FastAPI app — easy to run in one container.
- Includes GitHub Actions workflow to build & push to GHCR (uses `GITHUB_TOKEN` automatically, no PAT required).
- Optional API key to protect the endpoint.
- **Graceful fallback**: If Ollama fails, logs the error and returns documents in original order so RAG continues working.

## Quick start (Docker Compose)

```yaml
version: "3.8"
services:
  rerank-bridge:
    image: ghcr.io/beaudamore/openwebui-ollama-reranker-bridge:latest
    container_name: rerank-bridge
    restart: unless-stopped
    environment:
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
      - RERANK_MODE=embeddings
      - RERANK_PORT=5600
    ports:
      - "5600:5600"
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

Check health:
```bash
curl http://localhost:5600/health
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `RERANK_MODE` | `embeddings` | Mode: `embeddings` (recommended) or `generate` |
| `RERANK_PORT` | `5600` | Port to listen on |
| `RERANK_TIMEOUT` | `30` | Request timeout in seconds |
| `RERANK_BRIDGE_API_KEY` | (none) | Optional API key for authentication |
| `OLLAMA_API_KEY` | (none) | Optional Ollama API key |

## Deploy via GitHub Actions → GHCR → Portainer

1. Fork or clone this repo.
2. Push to main — the workflow automatically builds multi-arch (amd64+arm64) and pushes to GHCR.
3. In Portainer, create a stack referencing the image:
   ```
   ghcr.io/<your-user>/openwebui-ollama-reranker-bridge:latest
   ```
4. Ensure the bridge container can reach your Ollama instance (use `extra_hosts` or same Docker network).

## OpenWebUI Admin settings

- **Reranking Engine:** External
- **API Base URL:** `http://<bridge-host>:5600/v1/rerank`
- **Reranking Model:** Your Ollama model name (e.g., `xitao/bge-reranker-v2-m3:latest`)
- **API Key:** Set to `RERANK_BRIDGE_API_KEY` value if configured (optional)

## Notes on modes

- **`embeddings` (default, recommended):** Bridge calls Ollama embeddings for [query + docs], computes cosine similarity, normalizes 0..1, returns results. Deterministic and efficient.
- **`generate`:** Bridge prompts the model to produce a numeric score. Less deterministic, slower. Use only if running a cross-encoder reranker model that outputs scores via generation.

## Docker networking tips

The `extra_hosts` config maps `host.docker.internal` to your host machine, useful on Linux. On Docker Desktop (Mac/Windows) this works automatically.

If OpenWebUI and the bridge are in the same Docker network, you can use the container name directly:
```
http://rerank-bridge:5600/v1/rerank
```

## Security

Set `RERANK_BRIDGE_API_KEY` in the environment and put the same key in OpenWebUI's External API Key field (`Authorization: Bearer <key>`).
