from functools import lru_cache
from pathlib import Path

from pydantic import EmailStr, SecretStr, computed_field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        yaml_file="config.yaml",
        extra="ignore",
    )

    ncbi_api_key: str
    ncbi_email: EmailStr
    anthropic_api_key: SecretStr

    # cost-defense knobs
    monthly_cap_usd: float = 15.0
    per_query_ceiling_usd: float = 0.10
    max_tokens: int = 1024
    rerank_floor: float = 0.0

    hf_home: Path = Path("./data/hf_cache")
    data_dir: Path = Path("./data")

    @computed_field
    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "sqlite.db"

    @computed_field
    @property
    def faiss_index_path(self) -> Path:
        return self.data_dir / "faiss.index"

    @computed_field
    @property
    def faiss_chunk_ids_path(self) -> Path:
        return self.data_dir / "faiss.chunk_ids.json"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
