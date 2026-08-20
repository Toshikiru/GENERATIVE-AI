# Checkpoint 4 - Containerization
#
# Multi-stage build: dependencies (including torch, pulled in by
# sentence-transformers) are installed into an isolated prefix in the
# "builder" stage, then only that installed prefix + source code are
# copied into a clean runtime image, so build tools and pip's cache never
# end up in the final image.

FROM python:3.13-slim AS builder

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user --resume-retries 10 --timeout 100 -r requirements.txt

FROM python:3.13-slim

WORKDIR /app

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

COPY src/ src/
COPY data/raw/ data/raw/

# Bake the vector index into the image at build time (parses, chunks,
# embeds, and stores data/raw/*.txt into data/chroma_db/) so the container
# is immediately queryable on start -- no first-request cold-start ingest,
# no bind-mounted volume required for a basic run.
RUN python src/ingest.py

EXPOSE 8501

# GEMINI_API_KEY is required at runtime and is intentionally NOT set here --
# see the "Docker" section of README.md for how to pass it in via `docker
# run -e` / `docker-compose` secrets instead of baking it into the image.
CMD ["streamlit", "run", "src/web_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
