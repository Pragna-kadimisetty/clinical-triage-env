FROM python:3.11-slim
WORKDIR /app

# 1. Install build tools
RUN pip install --no-cache-dir setuptools wheel uv

# 2. Copy metadata files
COPY pyproject.toml uv.lock README.md ./

# 3. Copy your folders
COPY clinical_triage/ ./clinical_triage/
COPY server/ ./server/

# 4. Install everything
RUN pip install --no-cache-dir .

EXPOSE 7860

# 5. Run using the command we defined in pyproject.toml
CMD ["clinical-triage-server"]