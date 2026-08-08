# EdHome — Chatterbox Multilingual V3 (NAMUS KURULUMU)
# =====================================================
# Tedbirler:
# 1) torch YENİDEN KURULMAZ → multi-GB layer yazımı YOK (I/O fail kök nedeni)
# 2) gradio YOK (--no-deps) → image küçük
# 3) git var (perth/deps)
# 4) pip retry + uzun timeout
# 5) CACHE_BUST her sürümde değişir

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

ENV CACHE_BUST=NAMUS_V3_TR_EMO_FAST_003
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_DEFAULT_TIMEOUT=120
ENV CHATTERBOX_T3_MODEL=v3
ENV TTS_EXAGGERATION=0.7
ENV TTS_CFG_WEIGHT=0.3
ENV TTS_TEMPERATURE=0.8
ENV TTS_REPETITION_PENALTY=2.0
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

RUN echo "namus-$CACHE_BUST" > /tmp/cache_bust.txt \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      libsndfile1 \
      ffmpeg \
      git \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && python -m pip install --no-cache-dir --upgrade pip

COPY requirements.txt .

# Base image CUDA torch KALIR (2.4.x). chatterbox --no-deps.
# Bu tek karar, dünkü registry I/O felaketini engellemek için.
RUN pip install --no-cache-dir --retries 5 --timeout 120 \
      chatterbox-tts==0.1.7 --no-deps \
 && pip install --no-cache-dir --retries 5 --timeout 120 \
      -r requirements.txt \
 && python -c "import torch; print('TORCH', torch.__version__, 'CUDA', torch.cuda.is_available())"

COPY handler.py .
COPY zumrut_hoca.WAV .

# Build-time import smoke (GPU yok; sadece paket yolu)
RUN python -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES; assert 'tr' in SUPPORTED_LANGUAGES; print('CHATTERBOX_OK', sorted(SUPPORTED_LANGUAGES)[:5], '...')"

CMD ["python", "-u", "handler.py"]
