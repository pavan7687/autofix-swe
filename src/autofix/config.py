"""Central configuration for the training and evaluation pipeline.

Every hyperparameter, path and limit is a setting rather than a literal, so an
experiment is reproducible from its `.env` alone. Anything that changes a
result must be recorded; `Settings.fingerprint()` hashes the training-relevant
subset so a checkpoint can be traced back to the exact configuration that
produced it.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _csv(value: str | list[str], sep: str = ",") -> list[str]:
    if isinstance(value, list):
        return value
    return [item.strip() for item in value.split(sep) if item.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # --- paths ------------------------------------------------------------
    data_root: Path = Path("./artifacts/data")
    model_root: Path = Path("./artifacts/models")
    run_root: Path = Path("./artifacts/runs")
    workspace_root: Path = Path("./artifacts/work")

    # --- models -----------------------------------------------------------
    reranker_base: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    editor_base: str = "Qwen/Qwen2.5-Coder-32B-Instruct"
    editor_size_override: str | None = None
    max_seq_len_override: int | None = None

    # --- data -------------------------------------------------------------
    github_token: str | None = None
    eval_benchmarks: list[str] = Field(
        default_factory=lambda: [
            "princeton-nlp/SWE-bench_Lite",
            "princeton-nlp/SWE-bench_Verified",
        ]
    )
    retrieval_candidates: int = 50
    retrieval_keep: int = 5

    # --- training ---------------------------------------------------------
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    learning_rate: float = 1e-4
    num_epochs: int = 2
    warmup_ratio: float = 0.03
    grad_accum: int = 16
    seed: int = 42

    # --- rejection sampling ----------------------------------------------
    samples_per_issue: int = 8
    sampling_temperature: float = 0.8
    sampling_top_p: float = 0.95
    max_verify_concurrency: int = 4

    # --- serving ----------------------------------------------------------
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str = "EMPTY"

    # --- sandbox ----------------------------------------------------------
    docker_host: str = ""
    sandbox_network_mode: str = "none"
    sandbox_cpu_quota: float = 2.0
    sandbox_memory_mb: int = 4096
    sandbox_pids_limit: int = 512
    sandbox_timeout_seconds: int = 900
    sandbox_install_timeout_seconds: int = 600
    sandbox_allow_install_network: bool = True

    # --- patch scope ------------------------------------------------------
    max_files_changed: int = 5
    max_lines_changed: int = 200
    max_single_file_lines: int = 120
    protected_paths: list[str] = Field(default_factory=list)

    log_level: str = "INFO"
    log_json: bool = False

    @field_validator("eval_benchmarks", mode="before")
    @classmethod
    def _split_commas(cls, v: object) -> object:
        return _csv(v) if isinstance(v, str) else v

    @field_validator("protected_paths", mode="before")
    @classmethod
    def _split_semicolons(cls, v: object) -> object:
        return _csv(v, sep=";") if isinstance(v, str) else v

    @field_validator("editor_size_override")
    @classmethod
    def _check_size(cls, v: str | None) -> str | None:
        if v in (None, ""):
            return None
        if v.lower() not in {"7b", "14b", "32b"}:
            raise ValueError("EDITOR_SIZE_OVERRIDE must be 7b, 14b or 32b")
        return v.lower()

    # --- derived paths ----------------------------------------------------

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"

    @property
    def retrieval_dataset(self) -> Path:
        return self.data_root / "retrieval"

    @property
    def editing_dataset(self) -> Path:
        return self.data_root / "editing"

    @property
    def contamination_index(self) -> Path:
        return self.data_root / "contamination_index.json"

    def ensure_dirs(self) -> None:
        for path in (self.data_root, self.model_root, self.run_root, self.workspace_root):
            path.mkdir(parents=True, exist_ok=True)

    def fingerprint(self) -> str:
        """Stable hash of the settings that can change a training result."""
        relevant = {
            "reranker_base": self.reranker_base,
            "editor_base": self.editor_base,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "grad_accum": self.grad_accum,
            "seed": self.seed,
            "retrieval_candidates": self.retrieval_candidates,
            "retrieval_keep": self.retrieval_keep,
        }
        blob = json.dumps(relevant, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
