from pydantic import BaseModel
from typing import List

class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float