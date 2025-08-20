from pydantic import BaseModel, Field
from typing import Dict, Literal, Union, List

class Comment(BaseModel):
    song: str
    artist: str
    video_id: str
    text: str
