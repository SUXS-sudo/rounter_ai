from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


class Settings(BaseSettings):
    app_name: str = "MiMo 智能路线规划系统"
    app_version: str = "2.0.0"
    default_city: str = "成都"
    data_dir: Path = DATA_DIR
    pois_file: Path = DATA_DIR / "poi_data_500k.db"
    user_profiles_file: Path = DATA_DIR / "user_profiles.json"
    max_route_pois: int = Field(default=8, gt=0)
    default_route_duration_minutes: int = Field(default=240, gt=0)
    walking_speed_kmph: float = Field(default=4.5, gt=0)

    # LLM API configuration
    llm_provider: str = Field(
        default="anthropic",
        description="LLM provider: 'anthropic' for MiMo, 'openai' for DeepSeek/LongCat/etc.",
    )
    mimo_api_key: str = Field(default="", description="LLM API key")
    mimo_base_url: str = Field(
        default="https://api.siliconflow.cn/v1",
        description="LLM API base URL",
    )
    mimo_model: str = Field(
        default="XiaomiMiMo/MiMo-7B-RL",
        description="LLM model identifier",
    )
    mimo_temperature: float = Field(default=0.3, ge=0, le=2)
    mimo_max_tokens: int = Field(default=4096, gt=0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ROUTE_PLANNER_",
        extra="ignore",
    )


settings = Settings()
