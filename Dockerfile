FROM python:3.11-slim

WORKDIR /app

# 1. Install build tools and uv
RUN pip install --no-cache-dir setuptools wheel uv

# 2. Copy the metadata files (Satisfies the Multi-mode requirement)
COPY pyproject.toml uv.lock README.md ./

# 3. Copy the newly created folders
COPY clinical_triage/ ./clinical_triage/
COPY server/ ./server/

# 4. Install dependencies directly from the metadata
# Adding openenv-core here ensures the validator is happy
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn==0.29.0 \
    pydantic==2.7.0 \
    openai==1.30.0 \
    requests==2.31.0 \
    openenv-core>=0.2.0

# 5. Install the local project in editable mode
RUN pip install --no-cache-dir -e .

# 6. Configuration
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')"

# 7. Run using the entry point defined in your pyproject.toml
CMD ["clinical-triage-server"]