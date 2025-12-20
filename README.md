# OpenWebUI-Ollama-ReRanker-Bridge

A small, Docker-deployable bridge that implements the OpenWebUI external reranker contract and forwards rerank requests to an Ollama instance.

- POST /v1/rerank accepts:
  {
    "model": "<model-name>",
    "query": "<query-text>",
    "documents": ["doc0", "doc1", ...],
    "top_n": 3
  }

- Returns:
  {
    "results": [
      {"index": 0, "relevance_score": 0.92},
      {"index": 1, "relevance_score": 0.12},
      ...
    ]
  }

Design decisions
- Ollama-only (no public APIs). Uses Ollama embeddings endpoint by default (deterministic cosine similarity).
- Lightweight FastAPI app — easy to run in one container.
- Includes GitHub Actions workflow to build & push to GHCR so Portainer can pull the image.
- Optional API key to protect the endpoint.

Quick start (local / dev)
1. Copy the files into a new folder.
2. Copy `.env.example` to `.env` and update values (OLLAMA_BASE_URL, etc).
3. Build and run locally:
   docker compose up --build -d
4. Check health:
   curl http://localhost:8000/health

Deploy via GitHub Actions → GHCR → Portainer (recommended for production)
1. Create a new repo named `OpenWebUI-Ollama-ReRanker-Bridge` and push these files.
2. In repo Settings → Secrets → Actions add:
   - CR_PAT (or use GITHUB_TOKEN automatically for GHCR; see workflow notes)
3. Push to main (or manually run the workflow). The workflow builds multi-arch (amd64+arm64) and pushes to:
  - `ghcr.io/<github-username>/openwebui-ollama-reranker-bridge:latest`
  - `ghcr.io/<github-username>/openwebui-ollama-reranker-bridge:<git-sha>`
4. In Portainer create a stack referencing the pushed image:
  image: ghcr.io/<your-user>/openwebui-ollama-reranker-bridge:latest
5. Ensure the bridge container can reach your Ollama instance (same Docker network or use host IP).

Apple Silicon (M1/M2) / ARM64 notes
- The published image is now built for linux/amd64 and linux/arm64 via the GitHub Actions workflow. No `platform:` override is needed on macOS or Linux ARM.
- If you build locally instead of using GHCR, run a multi-arch build with buildx, e.g.:
  docker buildx build --platform linux/amd64,linux/arm64 -t ghcr.io/<user>/openwebui-ollama-reranker-bridge:latest --push .

OpenWebUI Admin settings
- Reranking Engine: External
- API Base URL: http://<bridge-host>:8000/v1/rerank
- Reranking Model: <model-name you want Ollama to use, e.g. xitao/bge-reranker-v2-m3>
- API Key: set to RERANK_BRIDGE_API_KEY if you configured it (optional)

Notes on modes
- Default mode (recommended): `embeddings` – bridge calls Ollama embeddings for [query + docs], computes cosine similarity, normalizes 0..1, returns results. Deterministic and efficient.
- Alternative mode `generate`: bridge prompts the model to produce a numeric score. Less deterministic, slower. Use only if running a cross-encoder reranker model that outputs scores via generation.

Security
- Set `RERANK_BRIDGE_API_KEY` in the environment and put same key in OpenWebUI External API Key UI field (Authorization: Bearer <key>).

Support & maintenance
- The bridge is small and independent; keep Ollama and the bridge in your Docker environment. If OpenWebUI changes in the future, nothing in OpenWebUI needs changing — the bridge accepts the one contract that OpenWebUI posts.

If you want, I will:
- Provide the exact Git commands to create the repo and push these files, OR
- Build & push the image to GHCR under a repo you give me (I cannot push to your account, but I can provide CI file so it builds on your repo automatically).
