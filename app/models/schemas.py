from pydantic import BaseModel
from typing import List

class ResumeEntity(BaseModel):
    label: str
    text: str

class ParseResponse(BaseModel):
    filename: str
    entities: List[ResumeEntity]
