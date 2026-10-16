# Case and Analysis models.
# Data lives in MongoDB; these are lightweight Python dataclass placeholders
# so that app/models/__init__.py imports resolve cleanly.

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Case:
    """A patient case grouping multiple dermoscopic visits."""
    _id: str = ""
    label: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Analysis:
    """One melanoma analysis tied to a Case."""
    _id: str = ""
    case_id: str = ""
    original_filename: Optional[str] = None
    stored_path: Optional[str] = None
    segmentation_path: Optional[str] = None
    is_valid: bool = True
    predicted_class: Optional[str] = None
    melanoma_probability: Optional[float] = None
    risk_level: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
