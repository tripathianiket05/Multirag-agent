FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY streamlit_app.py .
COPY run.py .
COPY .env.example .env.example
RUN mkdir -p data/documents data/images data/benchmarks
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]