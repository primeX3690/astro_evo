# Dockerfile - containerizes the AstroEvo REST API (module 12)
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn[standard]

COPY . .

EXPOSE 8000

WORKDIR /app/12_api
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
