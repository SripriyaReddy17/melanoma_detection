import uuid
from datetime import datetime

from sqlalchemy import Column, String, Float, Integer, Boolean, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from app.database import Base


def _uuid() -> str:
    return uuid.uuid4().hex


class Case(Base):
    """
    A case groups multiple images/analyses of the same lesion so that
    Evolution (E) can be computed across visits.
    """
    __tablename__ = "cases"

    id = Column(String(32), primary_key=True, default=_uuid)
    label = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    analyses = relationship(
        "Analysis",
        backref="case",
        order_by="Analysis.created_at",
        cascade="all, delete-orphan"
    )


class Analysis(Base):
    """
    One uploaded image and its full analysis result.
    Images are stored on disk; metadata + results in this table.
    """
    __tablename__ = "analyses"

    id = Column(String(32), primary_key=True, default=_uuid)
    case_id = Column(String(32), ForeignKey("cases.id"), nullable=True, index=True)

    original_filename = Column(String(512))
    stored_path = Column(String(1024))

    is_valid = Column(Boolean, default=True)
    image_type = Column(String(64))

    predicted_class = Column(String(64))
    melanoma_probability = Column(Float)
    risk_level = Column(String(16))

    asymmetry = Column(Float)
    border_irregularity = Column(Float)
    color_h = Column(Float)
    color_s = Column(Float)
    color_v = Column(Float)
    color_index = Column(Float)
    diameter_pixels = Column(Float)
    diameter_mm = Column(Float)

    evolution_available = Column(Boolean, default=False)
    evolution_change_detected = Column(Boolean, default=False)
    evolution_json = Column(Text)

    report_json = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)
