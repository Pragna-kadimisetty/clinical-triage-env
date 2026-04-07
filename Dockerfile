FROM python:3.11-slim

WORKDIR /app

# 1. Pre-install build tools to prevent 'setuptools.build-meta' errors
RUN pip install --no-cache-dir setuptools wheel

# 2. Copy the metadata files (Satisfies the OpenEnv Validator)
COPY pyproject.toml .
COPY README.md . 

# 3. Install dependencies directly
# This avoids the "BackendUnavailable" crash while keeping the repo compliant
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn==0.29.0 \
    pydantic==2.7.0 \
    openai==1.30.0 \
    requests==2.31.0

# 4. Copy the environment logic files
COPY models.py .
COPY patients.py .
COPY environment.py .
COPY app.py .

# 5. Standard Configuration
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')"

# 6. Run the server
CMD ["python", "app.py"]