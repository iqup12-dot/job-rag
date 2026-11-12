from fastapi import FastAPI
from pydantic import BaseModel

from pipeline import run

app = FastAPI(title="job-rag")


class AssignRequest(BaseModel):
    jd: str
    questions: list[str]


@app.post("/assign")
def assign(req: AssignRequest):
    return run(req.jd, req.questions)
