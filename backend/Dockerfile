FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/database/ database/
COPY backend/rag/ rag/
COPY backend/routers/ routers/
COPY backend/dependencies.py backend/schemas.py backend/main.py ./

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]