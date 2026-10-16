from pydantic import BaseModel
from typing import Optional

class AnalyzeRequest(BaseModel):
    patient_id: Optional[int] = None
    case_id: Optional[int] = None
