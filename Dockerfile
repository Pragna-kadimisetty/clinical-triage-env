FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn==0.29.0 \
    pydantic==2.7.0 \
    openai==1.30.0 \
    requests==2.31.0

COPY models.py .
COPY patients.py .
COPY environment.py .
COPY app.py .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')"

CMD ["python", "app.py"]