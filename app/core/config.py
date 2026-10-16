# ============================================================
# CEFM CONFIGURATION  —  pydantic-settings based
# ============================================================
#
# All runtime settings are loaded from environment variables
# and / or a `.env` file at the project root.
#
# Downstream code continues to import plain module-level
# constants (DATA_DIR, MONGODB_URL, …) exactly as before.
# ============================================================

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================
# PROJECT ROOT  (two levels up from  app/core/config.py)
# ============================================================

BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent


# ============================================================
# SETTINGS CLASS
# ============================================================

class Settings(BaseSettings):
    """
    Centralised, validated configuration.

    Values are read — in order of priority — from:
      1. Real environment variables
      2. A `.env` file in the project root
      3. The defaults declared here
    """

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",           # silently ignore unknown keys
        case_sensitive=False,
    )

    # --------------------------------------------------------
    # MongoDB
    # --------------------------------------------------------
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_database: str = "cefm_melanoma"

    # --------------------------------------------------------
    # DeepSeek / LLM API
    # --------------------------------------------------------
    deepseek_api_key: Optional[str] = None
    deepseek_api_url: Optional[str] = None
    deepseek_model: str = "deepseek-chat"

    # --------------------------------------------------------
    # Optional model directories
    # --------------------------------------------------------
    clip_model_dir: Optional[str] = None

    # --------------------------------------------------------
    # Upload safety limits
    # --------------------------------------------------------
    max_upload_size_mb: int = 25                  # 25 MiB default
    max_upload_bytes: Optional[int] = None        # derived or overridden
    allowed_extensions: str = ".jpg,.jpeg,.png"   # comma-separated

    def model_post_init(self, __context) -> None:
        if self.max_upload_bytes is None:
            self.max_upload_bytes = int(self.max_upload_size_mb * 1024 * 1024)
        elif "max_upload_size_mb" not in self.model_fields_set:
            self.max_upload_size_mb = max(1, self.max_upload_bytes // (1024 * 1024))


# Single shared instance — import `settings` elsewhere.
settings = Settings()


# ============================================================
# BACKWARD-COMPATIBLE MODULE-LEVEL CONSTANTS
# ============================================================
# Every import like  `from app.core.config import DATA_DIR`
# keeps working without any changes in the rest of the code.
# ============================================================

# --- MongoDB ------------------------------------------------
MONGODB_URL: str = settings.mongodb_url
MONGODB_DATABASE: str = settings.mongodb_database

# --- DeepSeek / LLM ----------------------------------------
DEEPSEEK_API_KEY: Optional[str] = settings.deepseek_api_key
DEEPSEEK_API_URL: Optional[str] = settings.deepseek_api_url
DEEPSEEK_MODEL: str = settings.deepseek_model

# --- Directories --------------------------------------------
DATA_DIR: Path = BASE_DIR / "data"
MODELS_DIR: Path = BASE_DIR / "models"
TEMP_DIR: Path = BASE_DIR / "temp"

ISIC2018_DIR: Path = DATA_DIR / "ISIC2018"
ISIC_IMAGES: Path = ISIC2018_DIR / "images"
ISIC_MASKS: Path = ISIC2018_DIR / "masks"
ISIC_METADATA: Path = ISIC2018_DIR / "metadata"

HAM10000_DIR: Path = DATA_DIR / "HAM10000"
HAM_IMAGES: Path = HAM10000_DIR / "images"
HAM_METADATA: Path = HAM10000_DIR / "metadata"

# --- Model directories --------------------------------------
CLASSIFICATION_MODEL_DIR: Path = MODELS_DIR / "classification"
SEGMENTATION_MODEL_DIR: Path = MODELS_DIR / "segmentation"
IMAGE_TYPE_MODEL_DIR: Path = MODELS_DIR / "image_type"

ULTRALIGHT_VMUNET_CHECKPOINT: Path = (
    SEGMENTATION_MODEL_DIR / "UltraLight_VM_UNet.pth"
)

CLIP_MODEL_DIR: Path = Path(
    settings.clip_model_dir
    or str(MODELS_DIR / "clip")
)

# --- Upload safety ------------------------------------------
MAX_UPLOAD_SIZE_MB: int = settings.max_upload_size_mb
MAX_UPLOAD_BYTES: int = settings.max_upload_bytes or (25 * 1024 * 1024)
ALLOWED_EXTENSIONS: set[str] = {
    ext.strip().lower()
    for ext in settings.allowed_extensions.split(",")
    if ext.strip()
}


# ============================================================
# SECRET / LOG SANITIZATION HELPERS
# ============================================================

import re

def sanitize_mongodb_url(url: str) -> str:
    """Mask credentials in MongoDB connection URI for safe logging."""
    if not url:
        return ""
    # Matches mongodb://user:pass@host or mongodb+srv://user:pass@host
    return re.sub(r"(mongodb(?:\+srv)?://)([^:@/]+):([^@/]+)@", r"\1\2:***@", str(url))


def sanitize_log_message(msg: str) -> str:
    """Mask any embedded passwords or connection strings in arbitrary log messages."""
    if not msg:
        return ""
    return sanitize_mongodb_url(str(msg))


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

for _dir in [
    DATA_DIR,
    MODELS_DIR,
    TEMP_DIR,
    ISIC_IMAGES,
    ISIC_MASKS,
    ISIC_METADATA,
    HAM_IMAGES,
    HAM_METADATA,
    CLASSIFICATION_MODEL_DIR,
    SEGMENTATION_MODEL_DIR,
    IMAGE_TYPE_MODEL_DIR,
    CLIP_MODEL_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# RUNTIME STATUS DICT  (unchanged public API)
# ============================================================

MODEL_RUNTIME_STATUS = {
    "ultralight_vmunet": {
        "model": "UltraLight VM-UNet",
        "checkpoint": str(ULTRALIGHT_VMUNET_CHECKPOINT),
        "checkpoint_exists": ULTRALIGHT_VMUNET_CHECKPOINT.is_file(),
    },
    "clip": {
        "env_var": "CLIP_MODEL_DIR",
        "path": str(CLIP_MODEL_DIR),
        "configured": bool(settings.clip_model_dir),
    },
    "deepseek": {
        "env_var": "DEEPSEEK_API_KEY",
        "api_url": DEEPSEEK_API_URL,
        "model": DEEPSEEK_MODEL,
        "configured": bool(DEEPSEEK_API_KEY and DEEPSEEK_API_URL),
    },
}
