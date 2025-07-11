from enum import Enum
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field

from webui.config import Config


class VoiceCombineRequest(BaseModel):
    """Request schema for voice combination endpoint that accepts either a string with + or a list"""

    voices: Union[str, List[str]] = Field(
        ...,
        description="Either a string with voices separated by + (e.g. 'voice1+voice2') or a list of voice names to combine",
    )


class TTSStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETED = "deleted"  # For files removed by cleanup


# OpenAI-compatible schemas
class WordTimestamp(BaseModel):
    """Word-level timestamp information"""

    word: str = Field(..., description="The word or token")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")


class CaptionedSpeechResponse(BaseModel):
    """Response schema for captioned speech endpoint"""

    audio: str = Field(..., description="The generated audio data encoded in base 64")
    audio_format: str = Field(..., description="The format of the output audio")
    timestamps: Optional[List[WordTimestamp]] = Field(
        ..., description="Word-level timestamps"
    )


class NormalizationOptions(BaseModel):
    """Options for the normalization system"""

    normalize: bool = Field(
        default=True,
        description="Normalizes input text to make it easier for the model to say",
    )
    unit_normalization: bool = Field(
        default=False, description="Transforms units like 10KB to 10 kilobytes"
    )
    url_normalization: bool = Field(
        default=True,
        description="Changes urls so they can be properly pronounced by kokoro",
    )
    email_normalization: bool = Field(
        default=True,
        description="Changes emails so they can be properly pronouced by kokoro",
    )
    optional_pluralization_normalization: bool = Field(
        default=True,
        description="Replaces (s) with s so some words get pronounced correctly",
    )
    phone_normalization: bool = Field(
        default=True,
        description="Changes phone numbers so they can be properly pronouced by kokoro",
    )
    replace_remaining_symbols: bool = Field(
        default=True,
        description="Replaces the remaining symbols after normalization with their words",
    )


class OpenAISpeechRequest(BaseModel):
    """Request schema for OpenAI-compatible speech endpoint"""

    model: str = Field(
        default=Config.model_name,
        description=f"Default model: {Config.model_name}, do not change it",
    )
    input: str = Field(..., description="The text to generate audio for")
    voice: str = Field(
        default="sft_中文女",
        description="""默认: sft_中文女, 其中中文女来自模型自带音色。
这个值由两部分组成，使用_连接：

1. 模式标识：可选值为 sft（模型自带音色）、clone（克隆音色）、crosslingual（跨语种克隆音色）
2. 音色标识：
- 仅当模式为 sft 时，可选值为模型自带音色列表（/v1/audio/voices_sft）里的音色。
- 仅当模式为 clone 时，可选值为预装音色列表（/v1/audio/voices）里的音色。
- 仅当模式为 crosslingual 时，可选值为预装音色列表（/v1/audio/voices）里的音色。

下版本添加支持用户上传音色音频""",
    )
    # response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
    response_format: Literal["wav"] = Field(
        default="wav",
        description="The format to return audio in. Supported format: wav, do not change it",
    )
    download_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]] = (
        Field(
            default=None,
            description="[Unused] Optional different format for the final download. If not provided, uses response_format.",
        )
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="The speed of the generated audio. Select a value from 0.25 to 4.0.",
    )
    stream: bool = Field(
        default=True,  # Default to streaming for OpenAI compatibility
        description="If true (default), audio will be streamed as it's generated. Each chunk will be a complete sentence.",
    )
    return_download_link: bool = Field(
        default=False,
        description="[Unused] If true, returns a download link in X-Download-Path header after streaming completes",
    )
    lang_code: Optional[str] = Field(
        default=None,
        description="[Unused] Optional language code to use for text processing. If not provided, will use first letter of voice name.",
    )
    volume_multiplier: Optional[float] = Field(
        default=Config.max_val,
        description="A volume multiplier to multiply the output audio by.",
    )
    normalization_options: Optional[NormalizationOptions] = Field(
        default=NormalizationOptions(),
        description="[Unused] Options for the normalization system",
    )


class CaptionedSpeechRequest(BaseModel):
    """[Unused] Request schema for captioned speech endpoint"""

    model: str = Field(
        default=Config.model_name,
        description=f"Default model: {Config.model_name}, do not change it",
    )
    input: str = Field(..., description="The text to generate audio for")
    voice: str = Field(
        default="sft_中文女",
        description="""默认: sft_中文女, 其中中文女来自模型自带音色。
这个值由两部分组成，使用_连接：

1. 模式标识：可选值为 sft（模型自带音色）、clone（克隆音色）、crosslingual（跨语种克隆音色）
2. 音色标识：
- 仅当模式为 sft 时，可选值为模型自带音色列表（/v1/audio/voices_sft）里的音色。
- 仅当模式为 clone 时，可选值为预装音色列表（/v1/audio/voices）里的音色。
- 仅当模式为 crosslingual 时，可选值为预装音色列表（/v1/audio/voices）里的音色。

下版本添加支持用户上传音色音频""",
    )
    # response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
    response_format: Literal["wav"] = Field(
        default="wav",
        description="The format to return audio in. Supported format: wav, do not change it",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="The speed of the generated audio. Select a value from 0.25 to 4.0.",
    )
    stream: bool = Field(
        default=True,  # Default to streaming for OpenAI compatibility
        description="If true (default), audio will be streamed as it's generated. Each chunk will be a complete sentence.",
    )
    return_timestamps: bool = Field(
        default=True,
        description="[Unused] If true (default), returns word-level timestamps in the response",
    )
    return_download_link: bool = Field(
        default=False,
        description="[Unused] If true, returns a download link in X-Download-Path header after streaming completes",
    )
    lang_code: Optional[str] = Field(
        default=None,
        description="[Unused] Optional language code to use for text processing. If not provided, will use first letter of voice name.",
    )
    volume_multiplier: Optional[float] = Field(
        default=1.0, description="A volume multiplier to multiply the output audio by."
    )
    normalization_options: Optional[NormalizationOptions] = Field(
        default=NormalizationOptions(),
        description="[Unused] Options for the normalization system",
    )
