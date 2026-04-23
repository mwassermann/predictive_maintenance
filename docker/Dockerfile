FROM python:3.10-slim

WORKDIR /app

# system deps (needed for many wheels)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# enforce critical constraints FIRST
RUN pip install --no-cache-dir \
    "tensorflow==2.10.1" \
    "protobuf<3.20" \
    && pip install --no-cache-dir -r requirements.txt


COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]