FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt requirements-onnx.txt ./
ARG INSTALL_ONNX=1
RUN pip install --no-cache-dir -r requirements.txt && \
    if [ "$INSTALL_ONNX" = "1" ]; then pip install --no-cache-dir -r requirements-onnx.txt; fi
COPY src ./src
RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
