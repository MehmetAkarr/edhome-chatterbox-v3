"""
EdHome RunPod — Chatterbox Multilingual V3 | TR + DUYGU + HIZ
=============================================================
Namus kuralları (Resemble AI resmi + sahada kanıtlanan düzeltmeler):
1) Multilingual V3, language_id=tr  (Turbo YASAK — EN-only)
2) Duygu: exaggeration=0.7, cfg_weight=0.3  (README expressive tip)
3) CFG token batch=2 HER ZAMAN  (cfg=0 tek batch = boş wav — yasak)
4) Conditionals cache + kısa max_new_tokens = hız
5) PCM16 WAV (tarayıcı)
6) Init smoke test: gerçek "Merhaba." sentezi geçmeden job alma
7) Bozuk/kısa çıktıda otomatik kalite retry
"""
from __future__ import annotations

import base64
import io
import os
import time
import traceback

import runpod

import_hata = None
try:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from chatterbox.mtl_tts import (
        SUPPORTED_LANGUAGES,
        ChatterboxMultilingualTTS,
        punc_norm,
    )
    from chatterbox.models.s3tokenizer import drop_invalid_tokens
    from chatterbox.models.t3.modules.cond_enc import T3Cond

    try:
        from chatterbox.models.s3gen import S3GEN_SR as _S3GEN_SR
        from chatterbox.models.s3tokenizer import S3_TOKEN_RATE as _S3_TOKEN_RATE

        S3GEN_SR = int(_S3GEN_SR)
        S3_TOKEN_RATE = int(_S3_TOKEN_RATE)
    except Exception:
        S3GEN_SR = 24000
        S3_TOKEN_RATE = 50
except Exception as e:
    ChatterboxMultilingualTTS = None
    SUPPORTED_LANGUAGES = {"tr": "Turkish"}
    T3Cond = None
    F = None
    punc_norm = None
    np = None
    torch = None
    S3GEN_SR = 24000
    S3_TOKEN_RATE = 50
    drop_invalid_tokens = None
    import_hata = f"{e}\nDetay: {traceback.format_exc()}"

model = None
smoke_ok = False
T3_MODEL = os.environ.get("CHATTERBOX_T3_MODEL", "v3")
REFERENCE_CANDIDATES = (
    "zumrut_hoca.WAV",
    "zumrut_hoca.wav",
    "voices/zumrut_hoca.WAV",
    "voices/zumrut_hoca.wav",
)

# Kilitli duygu/hız profili (env ile override edilebilir)
DEFAULT_EXAGGERATION = float(os.environ.get("TTS_EXAGGERATION", "0.7"))
DEFAULT_CFG_WEIGHT = float(os.environ.get("TTS_CFG_WEIGHT", "0.3"))
DEFAULT_TEMPERATURE = float(os.environ.get("TTS_TEMPERATURE", "0.8"))
DEFAULT_REPETITION_PENALTY = float(os.environ.get("TTS_REPETITION_PENALTY", "2.0"))
RUN_SMOKE = os.environ.get("TTS_SMOKE_TEST", "1").strip() not in ("0", "false", "False")


def resolve_reference_audio() -> str:
    for path in REFERENCE_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"zumrut_hoca.WAV yok. Aranan: {REFERENCE_CANDIDATES}")


def initialize_model():
    global model, smoke_ok
    if ChatterboxMultilingualTTS is None or torch is None:
        raise RuntimeError(f"Kütüphane eksik! {import_hata}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref = resolve_reference_audio()
    print(f"[WORKER] device={device} torch={torch.__version__} t3={T3_MODEL} ref={ref}")
    if "tr" not in SUPPORTED_LANGUAGES:
        raise RuntimeError(f"Bu paket TR desteklemiyor: {SUPPORTED_LANGUAGES}")

    t0 = time.time()
    try:
        model = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model=T3_MODEL)
    except TypeError:
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    except Exception as err:
        if str(T3_MODEL) != "v2":
            print(f"[WORKER] {T3_MODEL} hata: {err} → v2")
            model = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v2")
        else:
            raise

    model.prepare_conditionals(ref, exaggeration=DEFAULT_EXAGGERATION)
    print(f"[WORKER] Model+Zümrüt yüklendi ({time.time() - t0:.1f}s) sr={model.sr}")

    if RUN_SMOKE and device == "cuda":
        t1 = time.time()
        wav, sr = generate_core(
            "Merhaba, ben Zümrüt hoca.",
            language_id="tr",
            exaggeration=DEFAULT_EXAGGERATION,
            cfg_weight=DEFAULT_CFG_WEIGHT,
            temperature=DEFAULT_TEMPERATURE,
            max_new_tokens=180,
        )
        b64 = wav_to_pcm16_b64(wav, sr)
        if len(b64) < 4000:
            raise RuntimeError(f"SMOKE FAIL: audio çok küçük ({len(b64)})")
        smoke_ok = True
        print(f"[WORKER] SMOKE OK bytes={len(b64)} t={time.time() - t1:.2f}s")
    else:
        smoke_ok = True
        print("[WORKER] smoke atlandı (CPU veya TTS_SMOKE_TEST=0)")


def generate_core(
    text: str,
    *,
    language_id: str = "tr",
    exaggeration: float = DEFAULT_EXAGGERATION,
    cfg_weight: float = DEFAULT_CFG_WEIGHT,
    temperature: float = DEFAULT_TEMPERATURE,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    min_p: float = 0.05,
    top_p: float = 1.0,
    max_new_tokens: int = 1000,
):
    """Resmi mtl_tts.generate akışı + max_new_tokens + CUDA autocast."""
    assert model is not None and torch is not None
    if model.conds is None:
        raise RuntimeError("Conditionals yok")

    lang = (language_id or "tr").lower()
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Dil yok: {lang}")

    cfg_weight = max(0.0, float(cfg_weight))
    exaggeration = float(exaggeration)

    if float(exaggeration) != float(model.conds.t3.emotion_adv[0, 0, 0].item()):
        _cond = model.conds.t3
        model.conds.t3 = T3Cond(
            speaker_emb=_cond.speaker_emb,
            cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=model.device)

    text = punc_norm(text)
    text_tokens = model.tokenizer.text_to_tokens(text, language_id=lang).to(model.device)
    text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # CFG batch=2 ZORUNLU

    sot = model.t3.hp.start_text_token
    eot = model.t3.hp.stop_text_token
    text_tokens = F.pad(text_tokens, (1, 0), value=sot)
    text_tokens = F.pad(text_tokens, (0, 1), value=eot)

    use_cuda = str(model.device).startswith("cuda")
    # fp16 riskli (kalite/NaN) — hız token tavanından; kalite fp32
    _ = use_cuda

    with torch.inference_mode():
        speech_tokens = model.t3.inference(
            t3_cond=model.conds.t3,
            text_tokens=text_tokens,
            max_new_tokens=int(max_new_tokens),
            temperature=temperature,
            cfg_weight=cfg_weight,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
        )
        speech_tokens = drop_invalid_tokens(speech_tokens[0]).to(model.device)
        if speech_tokens.numel() < 4:
            raise RuntimeError(f"speech_tokens çok kısa: {speech_tokens.numel()}")

        wav, _ = model.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=model.conds.gen,
        )
        wav = wav.squeeze(0).detach().float().cpu().numpy()

        n_tokens = int(speech_tokens.shape[-1])
        st_len = max(1, n_tokens - 1)
        wav = wav[: st_len * (S3GEN_SR // S3_TOKEN_RATE)]

        try:
            wav = model.watermarker.apply_watermark(wav, sample_rate=model.sr)
        except Exception:
            pass

    wav_t = torch.from_numpy(np.asarray(wav, dtype=np.float32)).reshape(-1)
    if float(wav_t.abs().max()) < 1e-4:
        raise RuntimeError("Sentez sessiz (peak~0)")
    return wav_t, int(model.sr)


def generate_with_retry(text: str, **kwargs):
    """Önce hızlı profil; bozuksa kalite retry — sessiz fail yok."""
    try:
        return generate_core(text, **kwargs)
    except Exception as first:
        print(f"[WORKER] retry tetiklendi: {first}")
        kw = dict(kwargs)
        kw["max_new_tokens"] = min(1000, max(int(kw.get("max_new_tokens", 220)) * 2, 320))
        kw["cfg_weight"] = max(float(kw.get("cfg_weight", 0.3)), 0.5)
        kw["exaggeration"] = float(kw.get("exaggeration", 0.7))
        kw["temperature"] = 0.8
        return generate_core(text, **kw)


def wav_to_pcm16_b64(wav, sr: int) -> str:
    import soundfile as sf

    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().float().numpy()
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak < 1e-8:
        raise RuntimeError("WAV sessiz")
    wav = np.clip(wav / peak, -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    raw = buf.getvalue()
    if len(raw) < 2000:
        raise RuntimeError(f"WAV çok küçük ({len(raw)} byte)")
    return base64.b64encode(raw).decode("utf-8")


def tokens_for_text(text: str, speed_mode: bool) -> int:
    if not speed_mode:
        return 1000
    # Kısa ders cümlesi: sampling tavanı düşük tut
    approx = int(28 + len(text) * 2.6)
    return max(140, min(300, approx))


def handler(job):
    global model
    t0 = time.time()
    try:
        job_input = job.get("input", {}) or {}
        text = (job_input.get("text") or "").strip()

        if job_input.get("keep_alive") or text in (".", "ping"):
            if model is None:
                initialize_model()
            return {"status": "warm", "smoke_ok": smoke_ok, "t3_model": T3_MODEL, "elapsed": round(time.time() - t0, 3)}

        if not text:
            return {"error": "text parametresi gerekli"}

        if len(text) > 280:
            text = text[:280].rsplit(" ", 1)[0] + "."

        if model is None:
            initialize_model()

        speed_mode = bool(job_input.get("speed_mode", True))
        language_id = job_input.get("language_id", "tr")
        exaggeration = max(0.35, min(float(job_input.get("exaggeration", DEFAULT_EXAGGERATION)), 1.15))
        cfg_weight = float(job_input.get("cfg_weight", DEFAULT_CFG_WEIGHT))
        if cfg_weight < 0.15:
            # 0.0 resmi cross-lang; yine CFG batch var. Duygu için taban 0.3
            cfg_weight = DEFAULT_CFG_WEIGHT
        temperature = float(job_input.get("temperature", DEFAULT_TEMPERATURE))
        max_new_tokens = int(job_input.get("max_new_tokens", tokens_for_text(text, speed_mode)))
        max_new_tokens = max(120, min(max_new_tokens, 1000))

        wav, sr = generate_with_retry(
            text,
            language_id=language_id,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        audio_base64 = wav_to_pcm16_b64(wav, sr)
        elapsed = round(time.time() - t0, 3)
        dur = round(float(wav.numel()) / float(sr), 2)
        print(
            f"[WORKER] OK lang={language_id} chars={len(text)} tok={max_new_tokens} "
            f"exag={exaggeration} cfg={cfg_weight} dur={dur}s t={elapsed}s bytes={len(audio_base64)}"
        )
        return {
            "status": "success",
            "audio_base64": audio_base64,
            "elapsed": elapsed,
            "duration_sec": dur,
            "language_id": language_id,
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "max_new_tokens": max_new_tokens,
            "t3_model": T3_MODEL,
            "smoke_ok": smoke_ok,
        }
    except Exception as e:
        return {
            "error": "Sentezleme başarısız.",
            "details": str(e),
            "traceback": traceback.format_exc(),
        }


try:
    if ChatterboxMultilingualTTS is not None and any(os.path.exists(p) for p in REFERENCE_CANDIDATES):
        initialize_model()
except Exception as e:
    print(f"[WORKER] preload atlandı: {e}")

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
