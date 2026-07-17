FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

ENV MAGENT_LLM_PROVIDER=mock \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "magent.api:app", "--host", "0.0.0.0", "--port", "8000"]
