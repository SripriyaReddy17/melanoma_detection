from pydantic import BaseModel
from typing import Optional, Dict, Any

class ImageValidationResult(BaseModel):
    valid: bool
    reason: str
class ImageTypeResult(BaseModel):
    image_type: str
    confidence: float
    supported: bool
    message: str
    
class ABCDEResult(BaseModel):
    asymmetry: Optional[float] = None
    border_irregularity: Optional[float] = None
    color_h: Optional[float] = None
    color_s: Optional[float] = None
    color_v: Optional[float] = None
    diameter_pixels: Optional[float] = None
    diameter_mm: Optional[float] = None
    evolution: Optional[Dict[str, Any]] = None

class AnalyzeResponse(BaseModel):
    validation: ImageValidationResult
    message: str
