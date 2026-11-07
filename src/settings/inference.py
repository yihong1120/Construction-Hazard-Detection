from __future__ import annotations

from typing import Annotated

from pydantic import AliasChoices
from pydantic import Field
from pydantic import field_validator
from pydantic_settings import NoDecode

from src.settings.base import EnvironmentSettings


class ApiInferenceSettings(EnvironmentSettings):
    max_decoded_bytes: int = Field(
        256 * 1024 * 1024,
        ge=1,
        validation_alias='DETECT_MAX_DECODED_BYTES',
    )
    concurrency: int = Field(
        4,
        ge=1,
        le=32,
        validation_alias='YOLO_API_INFERENCE_CONCURRENCY',
    )
    queue_capacity: int = Field(
        8,
        ge=1,
        le=128,
        validation_alias=AliasChoices(
            'YOLO_API_QUEUE_CAPACITY',
            'DETECT_INGRESS_CONCURRENCY',
        ),
    )
    max_upload_bytes: int = Field(
        20 * 1024 * 1024,
        ge=1,
        validation_alias='DETECT_MAX_UPLOAD_BYTES',
    )
    max_image_pixels: int = Field(
        64 * 1024 * 1024,
        ge=1,
        validation_alias='DETECT_MAX_IMAGE_PIXELS',
    )
    use_tensorrt: bool = False
    use_sahi: bool = False
    max_loaded_models: int = Field(3, ge=1)
    device: str = Field(
        'auto', validation_alias=AliasChoices(
            'YOLO_INFERENCE_DEVICE', 'YOLO_DEVICE',
        ),
    )

    @field_validator('device')
    @classmethod
    def device_name(cls, value: str) -> str:
        return value.lower()


class WorkerSettings(EnvironmentSettings):
    model_config = {'env_prefix': 'YOLO_WORKER_'}

    timeout_seconds: float = Field(30, gt=0, allow_inf_nan=False)
    ring_slots: int = Field(2, ge=1)
    ring_slot_cleanup_seconds: float = Field(120.0, ge=0)
    device: str = 'cuda:0'
    precision: str | None = None
    model_dir: str = 'models/pt'
    model_suffix: str = '.pt'
    half: bool = True
    quantize: str | None = None
    batch_size: int = Field(8, ge=1)
    batch_wait_ms: int = Field(10, ge=0)
    imgsz: int = Field(640, ge=1)
    metrics_interval_seconds: float = 0
    artifact_manifest: str | None = None

    @field_validator('metrics_interval_seconds', mode='before')
    @classmethod
    def optional_metrics_interval(cls, value: object) -> float:
        try:
            return max(0.0, float(str(value)))
        except ValueError:
            return 0.0


class WorkerPoolSettings(EnvironmentSettings):
    model_config = {'env_prefix': 'YOLO_WORKER_'}
    cameras_per_engine: int = Field(3, ge=1)
    cameras_per_engine_by_model: Annotated[
        dict[str, int],
        NoDecode,
    ] = Field(default_factory=dict)
    queue_size: int = Field(64, ge=1)
    result_queue_size: int = Field(8, ge=1)
    devices: Annotated[tuple[str, ...], NoDecode] = ('cuda:0',)

    @field_validator('devices', mode='before')
    @classmethod
    def device_list(cls, value):
        if isinstance(value, str):
            value = tuple(
                item.strip() for item in value.split(',') if item.strip()
            )
        if not value:
            raise ValueError('At least one worker device is required')
        return value

    @field_validator('cameras_per_engine_by_model', mode='before')
    @classmethod
    def model_limits(cls, value):
        if isinstance(value, str):
            value = dict(
                item.strip().split(
                    '=',
                    1,
                )
                for item in value.split(',')
                if item.strip()
            )
        result = {key.lower(): int(limit) for key, limit in value.items()}
        if any(limit < 1 for limit in result.values()):
            raise ValueError('Worker camera limits must be positive')
        return result


class LocalInferenceSettings(EnvironmentSettings):
    model_config = {'env_prefix': 'DETECT_LOCAL_'}
    device: str = Field('cuda:0', min_length=1)
    imgsz: int = Field(640, ge=1)
    half: bool = True
    quantize: str | None = None


class ModelCatalogSettings(EnvironmentSettings):
    upload_max_bytes: int = Field(
        6 * 1024**3,
        ge=1,
        validation_alias='MODEL_UPLOAD_MAX_BYTES',
    )
    server_model_keys: str = Field(
        'yolo26n,yolo26s,yolo26m,yolo26l,yolo26x',
        validation_alias='DETECT_SERVER_MODEL_KEYS',
    )

    @property
    def model_keys(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                key.strip()
                for key in self.server_model_keys.split(',')
                if key.strip()
            ),
        )


class WebSocketSettings(EnvironmentSettings):
    max_session_seconds: float = Field(
        1800,
        gt=0,
        allow_inf_nan=False,
        validation_alias='WS_MAX_SESSION_SECONDS',
    )
    allow_localhost_bypass: bool = Field(
        True,
        validation_alias='YOLO_WS_ALLOW_LOCALHOST_BYPASS',
    )
    local_username: str = Field(
        'local-main',
        min_length=1,
        validation_alias='YOLO_WS_LOCAL_USERNAME',
    )
