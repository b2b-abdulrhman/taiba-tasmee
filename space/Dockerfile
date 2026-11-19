FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl bzip2 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Meta's Omnilingual ASR (CTC, 300M, int8) packaged by sherpa-onnx
RUN mkdir -p /models && curl -fsSL -o /tmp/m.tar.bz2 \
      https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12.tar.bz2 \
 && tar xjf /tmp/m.tar.bz2 -C /models && rm /tmp/m.tar.bz2 \
 && mv /models/sherpa-onnx-omnilingual-* /models/omni && rm -rf /models/omni/test_wavs
COPY --chown=user . /app
USER user
ENV MODEL_DIR=/models/omni NUMBA_CACHE_DIR=/tmp/numba THREADS=2 PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--ws-max-size", "4194304"]
