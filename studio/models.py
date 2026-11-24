"""Every weight file the studio uses: where it comes from and where it lives on disk.

Files are stored flat under MODELS_DIR/<group>/<basename>. `scripts/download_models.py`
reads this list; the engines look files up by key with `path(key)`.
"""

from dataclasses import dataclass
from pathlib import Path

from .config import MODELS_DIR


@dataclass(frozen=True)
class File:
    key: str
    group: str
    size: int  # bytes, as published; used for progress and disk checks
    repo: str = ""  # Hugging Face repo id
    path: str = ""  # path inside the repo
    url: str = ""  # direct download, for the few files not on Hugging Face
    license: str = ""

    @property
    def filename(self) -> str:
        return Path(self.path or self.url).name

    @property
    def local(self) -> Path:
        return MODELS_DIR / self.group / self.filename


FILES: list[File] = [
    # --- transcription: whisper.cpp quantized (GGML q5_0) ---
    File("whisper-large-v3-turbo-q5", "whisper", 574_041_195, "ggerganov/whisper.cpp", "ggml-large-v3-turbo-q5_0.bin", license="MIT"),
    File("whisper-large-v3-q5", "whisper", 1_081_140_203, "ggerganov/whisper.cpp", "ggml-large-v3-q5_0.bin", license="MIT"),
    File("silero-vad", "whisper", 885_098, "ggml-org/whisper-vad", "ggml-silero-v5.1.2.bin", license="MIT"),
    # --- translation ---
    File("translategemma-4b", "translate", 2_489_909_760, "mradermacher/translategemma-4b-it-GGUF", "translategemma-4b-it.Q4_K_M.gguf", license="Gemma"),
    # --- voice ---
    File("kokoro", "voice/kokoro", 325_532_232, "onnx-community/Kokoro-82M-v1.0-ONNX", "onnx/model.onnx", license="Apache-2.0"),
    # Kokoro voices are fetched as a folder; see VOICE_DIRS below.
    # The language model runs on the CPU: DirectML can't execute its GroupQueryAttention nodes.
    # The 4-bit export is the one built for fast CPU inference.
    File("chatterbox-lm", "voice/chatterbox", 227_911, "onnx-community/chatterbox-multilingual-ONNX", "onnx/language_model_q4.onnx", license="MIT"),
    File("chatterbox-lm-data", "voice/chatterbox", 353_621_248, "onnx-community/chatterbox-multilingual-ONNX", "onnx/language_model_q4.onnx_data", license="MIT"),
    File("chatterbox-decoder", "voice/chatterbox", 6_350_448, "onnx-community/chatterbox-multilingual-ONNX", "onnx/conditional_decoder.onnx", license="MIT"),
    File("chatterbox-decoder-data", "voice/chatterbox", 533_970_816, "onnx-community/chatterbox-multilingual-ONNX", "onnx/conditional_decoder.onnx_data", license="MIT"),
    File("chatterbox-embed", "voice/chatterbox", 13_286, "onnx-community/chatterbox-multilingual-ONNX", "onnx/embed_tokens.onnx", license="MIT"),
    File("chatterbox-embed-data", "voice/chatterbox", 68_390_912, "onnx-community/chatterbox-multilingual-ONNX", "onnx/embed_tokens.onnx_data", license="MIT"),
    File("chatterbox-speech-encoder", "voice/chatterbox", 1_184_608, "onnx-community/chatterbox-multilingual-ONNX", "onnx/speech_encoder.onnx", license="MIT"),
    File("chatterbox-speech-encoder-data", "voice/chatterbox", 591_274_880, "onnx-community/chatterbox-multilingual-ONNX", "onnx/speech_encoder.onnx_data", license="MIT"),
    File("chatterbox-tokenizer", "voice/chatterbox", 71_798, "onnx-community/chatterbox-multilingual-ONNX", "tokenizer.json", license="MIT"),
    File("chatterbox-tokenizer-config", "voice/chatterbox", 244, "onnx-community/chatterbox-multilingual-ONNX", "tokenizer_config.json", license="MIT"),
    File("chatterbox-cangjie", "voice/chatterbox", 1_920_163, "onnx-community/chatterbox-multilingual-ONNX", "Cangjie5_TC.json", license="MIT"),
    File("chatterbox-default-voice", "voice/chatterbox", 714_320, "onnx-community/chatterbox-multilingual-ONNX", "default_voice.wav", license="MIT"),
    # --- music: ACE-Step 1.5 for acestep.cpp ---
    File("ace-text-encoder", "music", 784_144_960, "Serveurperso/ACE-Step-1.5-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf", license="Apache-2.0"),
    File("ace-lm", "music", 1_975_837_568, "Serveurperso/ACE-Step-1.5-GGUF", "acestep-5Hz-lm-1.7B-Q8_0.gguf", license="Apache-2.0"),
    File("ace-dit", "music", 1_970_472_064, "Serveurperso/ACE-Step-1.5-GGUF", "acestep-v15-turbo-Q6_K.gguf", license="Apache-2.0"),
    File("ace-vae", "music", 337_420_928, "Serveurperso/ACE-Step-1.5-GGUF", "vae-BF16.gguf", license="Apache-2.0"),
    # --- image: shared FLUX VAE (Z-Image, FLUX Krea, Kontext) ---
    File("flux-vae", "image/shared", 335_304_388, "Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors", license="Apache-2.0"),
    # Z-Image-Turbo
    File("z-image-turbo", "image/z-image", 3_864_250_304, "leejet/Z-Image-Turbo-GGUF", "z_image_turbo-Q4_K.gguf", license="Apache-2.0"),
    File("qwen3-4b", "image/z-image", 2_497_281_120, "unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf", license="Apache-2.0"),
    # Qwen-Image 2.1 (+ Turbo): generation and instruction editing
    File("qwen-image", "image/qwen-image", 4_197_494_816, "leejet/Qwen-Image-2.1-GGUF", "qwen_image_2.1-Q4_K.gguf", license="Apache-2.0"),
    File("qwen-image-turbo", "image/qwen-image", 4_199_565_120, "unsloth/Qwen-Image-2.1-Turbo-GGUF", "qwen-image-2.1-turbo-Q4_K_M.gguf", license="Apache-2.0"),
    File("qwen-image-vae", "image/qwen-image", 675_509_688, "Comfy-Org/Qwen-Image-2.1", "vae/qwen_image_2.1_vae_bf16.safetensors", license="Apache-2.0"),
    File("qwen3-vl-8b", "image/qwen-image", 5_027_784_800, "Qwen/Qwen3-VL-8B-Instruct-GGUF", "Qwen3VL-8B-Instruct-Q4_K_M.gguf", license="Apache-2.0"),
    File("qwen3-vl-8b-vision", "image/qwen-image", 752_289_728, "Qwen/Qwen3-VL-8B-Instruct-GGUF", "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf", license="Apache-2.0"),
    # FLUX.1 Krea / Kontext (non-commercial FLUX.1 [dev] licence)
    File("flux-krea", "image/flux", 6_797_337_888, "QuantStack/FLUX.1-Krea-dev-GGUF", "flux1-krea-dev-Q4_K_S.gguf", license="FLUX.1-dev non-commercial"),
    File("flux-kontext", "image/flux", 6_797_337_888, "QuantStack/FLUX.1-Kontext-dev-GGUF", "flux1-kontext-dev-Q4_K_S.gguf", license="FLUX.1-dev non-commercial"),
    File("t5xxl", "image/flux", 2_896_123_072, "city96/t5-v1_1-xxl-encoder-gguf", "t5-v1_1-xxl-encoder-Q4_K_M.gguf", license="Apache-2.0"),
    File("clip-l", "image/flux", 246_144_152, "comfyanonymous/flux_text_encoders", "clip_l.safetensors", license="MIT"),
    # Photo-Mate adapters for Kontext
    File("lora-photo-restore", "image/flux/loras", 613_000_000, "prithivMLmods/Photo-Restore-i2i", "Photo-Restore-i2i.safetensors", license="FLUX.1-dev non-commercial"),
    File("lora-unblur-upscale", "image/flux/loras", 613_000_000, "prithivMLmods/Kontext-Unblur-Upscale", "Kontext-Image-Upscale.safetensors", license="FLUX.1-dev non-commercial"),
    File("lora-watermark-remover", "image/flux/loras", 613_000_000, "prithivMLmods/Kontext-Watermark-Remover", "Kontext-Watermark-Remover.safetensors", license="Apache-2.0 adapter on FLUX.1-dev"),
    File("lora-photo-cleanser", "image/flux/loras", 613_000_000, "prithivMLmods/PhotoCleanser-i2i", "PhotoCleanser-i2i.safetensors", license="FLUX.1-dev non-commercial"),
    # Stable Diffusion XL, quantized (Q8_0 GGUF): generation + inpainting (used for outpainting)
    File("sdxl", "image/sdxl", 5_040_957_760, "gpustack/stable-diffusion-xl-base-1.0-GGUF", "stable-diffusion-xl-base-1.0-Q8_0.gguf", license="CreativeML Open RAIL++-M"),
    File("sdxl-inpaint", "image/sdxl", 4_873_718_560, "gpustack/stable-diffusion-xl-inpainting-1.0-GGUF", "stable-diffusion-xl-inpainting-1.0-Q8_0.gguf", license="CreativeML Open RAIL++-M"),
    File("sdxl-vae", "image/sdxl", 334_641_162, "madebyollin/sdxl-vae-fp16-fix", "sdxl_vae.safetensors", license="MIT"),
    # Upscaling and background removal
    File("realesrgan-x4", "image/upscale", 67_040_989, url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth", license="BSD-3-Clause"),
    File("birefnet", "image/birefnet", 972_666_916, "onnx-community/BiRefNet-ONNX", "onnx/model.onnx", license="MIT"),
    # --- animation: Wan ---
    File("umt5-xxl", "video/shared", 3_655_145_312, "city96/umt5-xxl-encoder-gguf", "umt5-xxl-encoder-Q4_K_M.gguf", license="Apache-2.0"),
    File("wan21-vae", "video/shared", 253_815_318, "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/vae/wan_2.1_vae.safetensors", license="Apache-2.0"),
    File("wan21-t2v-1.3b", "video/wan21", 1_535_768_800, "samuelchristlie/Wan2.1-T2V-1.3B-GGUF", "Wan2.1-T2V-1.3B-Q8_0.gguf", license="Apache-2.0"),
    File("wan22-ti2v-5b", "video/wan22-5b", 3_433_116_000, "QuantStack/Wan2.2-TI2V-5B-GGUF", "Wan2.2-TI2V-5B-Q4_K_M.gguf", license="Apache-2.0"),
    File("wan22-vae", "video/wan22-5b", 1_409_400_960, "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/vae/wan2.2_vae.safetensors", license="Apache-2.0"),
    File("wan22-i2v-high", "video/wan22-14b", 8_757_992_064, "jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF", "high_noise/wan2.2_i2v_A14b_high_noise_lightx2v_4step-Q4_K_S.gguf", license="Apache-2.0"),
    File("wan22-i2v-low", "video/wan22-14b", 8_757_992_064, "jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF", "low_noise/wan2.2_i2v_A14b_low_noise_lightx2v_4step-Q4_K_S.gguf", license="Apache-2.0"),
]

# Folders downloaded whole: (repo, glob inside repo, group)
VOICE_DIRS = [("onnx-community/Kokoro-82M-v1.0-ONNX", "voices/*.bin", "voice/kokoro/voices")]

BY_KEY = {f.key: f for f in FILES}


def path(key: str) -> Path:
    return BY_KEY[key].local


def present(*keys: str) -> bool:
    return all(BY_KEY[k].local.exists() for k in keys)


def missing(*keys: str) -> list[str]:
    return [k for k in keys if not BY_KEY[k].local.exists()]
