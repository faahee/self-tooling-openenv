FROM python:3.11-slim

WORKDIR /app

COPY requirements-openenv.txt /app/requirements-openenv.txt
RUN pip install --no-cache-dir -r /app/requirements-openenv.txt

COPY support_triage_env /app/support_triage_env
COPY openenv.yaml /app/openenv.yaml
COPY README.md /app/README.md
COPY inference.py /app/inference.py

ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "support_triage_env.server:app", "--host", "0.0.0.0", "--port", "7860"]
