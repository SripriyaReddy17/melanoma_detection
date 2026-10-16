"""
CEFM Database Module — MongoDB + legacy SQLite.

MongoDB connection failures are handled gracefully so that
the application can start even when the database is
unreachable (health-check will report the problem).
"""

from pathlib import Path

# ============================================================
# LEGACY SQLITE  (retained for Streamlit frontend history)
# ============================================================
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_URL = f"sqlite:///{BASE_DIR / 'cefm.db'}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================================
# MONGODB CONNECTION
# ============================================================

from pymongo import MongoClient
from pymongo.errors import (
    ConnectionFailure,
    ServerSelectionTimeoutError,
)
from app.core.config import (
    MONGODB_URL,
    MONGODB_DATABASE,
    sanitize_log_message,
)


client = MongoClient(
    MONGODB_URL,
    serverSelectionTimeoutMS=5000,
)

db = client[MONGODB_DATABASE]


# ============================================================
# COLLECTIONS
# ============================================================

cases_collection = db["cases"]
analyses_collection = db["analyses"]
rag_documents_collection = db["rag_documents"]


# ============================================================
# CONNECTION CHECK  (safe — never raises)
# ============================================================

def check_mongodb() -> bool:
    """
    Return *True* when MongoDB is reachable, *False* otherwise.

    This intentionally swallows all exceptions so that callers
    (startup events, health endpoints) can degrade gracefully.
    """
    try:
        client.admin.command("ping")
        return True
    except (ConnectionFailure, ServerSelectionTimeoutError):
        return False
    except Exception:
        return False


# ============================================================
# SAFE WRITE HELPERS
# ============================================================
# Wrap common write operations so that the API route layer
# can call them without scattering try/except blocks for
# every single Mongo operation.
# ============================================================

def safe_insert_one(collection, document: dict) -> bool:
    """
    Insert *document* into *collection*.

    Returns ``True`` on success, ``False`` on any MongoDB error.
    Errors are printed to stderr for debugging (with credentials masked)
    but are never propagated — the caller decides how to react.
    """
    try:
        collection.insert_one(document)
        return True
    except Exception as exc:
        safe_msg = sanitize_log_message(str(exc))
        print(f"[MongoDB ERROR] insert_one failed: {type(exc).__name__}: {safe_msg}")
        return False


def safe_find_one(collection, query: dict, **kwargs):
    """
    Return a single document or ``None`` on failure.

    On a connection error the function returns ``None`` —
    identical to "no document found" — so the calling code
    follows its normal empty-result path.
    """
    try:
        return collection.find_one(query, **kwargs)
    except Exception as exc:
        safe_msg = sanitize_log_message(str(exc))
        print(f"[MongoDB ERROR] find_one failed: {type(exc).__name__}: {safe_msg}")
        return None