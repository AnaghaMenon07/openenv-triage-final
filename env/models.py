from pydantic import BaseModel


class Action(BaseModel):
    category: str
    priority: str
    response: str