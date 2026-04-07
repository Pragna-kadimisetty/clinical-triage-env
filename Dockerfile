FROM python:3.11-slim

WORKDIR /app

# 1. Copy the pyproject.toml first to leverage Docker cache
COPY pyproject.toml .

# 2. Install the dependencies and the project in editable mode
# This satisfies the "multi-mode deployment" requirement
RUN pip install --no-cache-dir .

# 3. Copy the rest of your application code
COPY models.py .
COPY patients.py .
COPY environment.py .
COPY app.py .

# 4. Expose the port required by Hugging Face
EXPOSE 7860

# 5. Standard Healthcheck to ensure the server is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')"

# 6. Run the application
CMD ["python", "app.py"]