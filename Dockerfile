FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs ./configs
COPY data/legal_docs ./data/legal_docs
COPY scripts ./scripts
COPY src ./src
COPY ui ./ui

EXPOSE 7860

CMD ["python", "ui/app.py"]
