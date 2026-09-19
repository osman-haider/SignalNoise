FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Generate the synthetic corpus/eval set at build time (deterministic, no
# network needed). Corpus embeddings are NOT built here on purpose -- that
# needs OFFICIAL_OPENAI_API_KEY, which should be a runtime secret, not a build-time
# one. Run scripts/build_embeddings.py once after the container is up
# (e.g. via a one-off `docker exec`), or on first request the app will
# simply run in degraded (lexical-only) hybrid mode until you do.
RUN python scripts/generate_corpus.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
