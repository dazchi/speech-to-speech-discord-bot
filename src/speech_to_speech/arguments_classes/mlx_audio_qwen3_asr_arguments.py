from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MLXAudioQwen3ASRSTTHandlerArguments:
    mlx_audio_qwen3_asr_model_name: str = field(
        default="mlx-community/Qwen3-ASR-1.7B-8bit",
        metadata={
            "help": (
                "The MLX Qwen3-ASR checkpoint to load through mlx-audio. "
                "'mlx-community/Qwen3-ASR-0.6B-8bit' is faster and less accurate. "
                "Default is 'mlx-community/Qwen3-ASR-1.7B-8bit'."
            )
        },
    )
    language: Optional[str] = field(
        default="auto",
        metadata={
            "help": (
                "The language of the speech as an ISO code ('en', 'zh', ...) or a name ('Chinese'), "
                "or 'auto' to identify it per utterance. Default is 'auto'."
            )
        },
    )
    mlx_audio_qwen3_asr_prompt: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional context or hotwords that bias the transcription, "
                "for example 'Alex, 艾丽同学, Discord'. Default is None."
            )
        },
    )
    mlx_audio_qwen3_asr_gen_max_new_tokens: int = field(
        default=256,
        metadata={"help": "The maximum number of tokens to generate per turn. Default is 256."},
    )
