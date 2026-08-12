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
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# pydantic-settings JSON-decodes any complex field (list, dict, set) read from
# the environment or a .env file, and it does so INSIDE the settings source -
# before any `field_validator(mode="before")` can run. So a human-friendly
# `EVAL_BENCHMARKS=a,b` blows up with a JSONDecodeError that names the field but
# not the cause.
#
# `NoDecode` disables that automatic decoding for a field, handing the raw
# string to our validators instead. The alternative - forcing operators to write
# `EVAL_BENCHMARKS=["a","b"]` in .env - is hostile for a file people edit by
# hand, and the semicolon-separated glob list would be worse still.
CsvList = Annotated[list[str], NoDecode]


def _csv(value: str | list[str], sep: str = ",") -> list[str]:
    """Parse a delimited string into a list, tolerating JSON as well.

    With NoDecode in play we own the parsing entirely, so we accept both the
    hand-written form (`a,b,c`) and the JSON form (`["a","b"]`) rather than
    silently mangling the latter into a single-element list.
    """
    if isinstance(value, list):
        return value
    text = value.strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in text.split(sep) if item.strip()]


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
    eval_benchmarks: CsvList = Field(
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
    protected_paths: CsvList = Field(default_factory=list)

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

    @field_validator(
        "editor_size_override", "max_seq_len_override", "github_token",
        mode="before",
    )
    @classmethod
    def _blank_is_none(cls, v: object) -> object:
        """Treat an empty .env value as unset.

        `MAX_SEQ_LEN_OVERRIDE=` is the natural way to write "no override" in a
        .env file, but pydantic hands the empty string straight to the int
        parser and it fails with a message that does not hint at the cause.
        Optional settings must accept the blank form.
        """
        if isinstance(v, str) and not v.strip():
            return None
        return v

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
