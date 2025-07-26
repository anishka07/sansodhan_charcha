from pydantic import BaseModel


class RAGResult(BaseModel):
    response: str = ""
    