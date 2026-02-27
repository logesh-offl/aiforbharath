import asyncio
import os
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl

from agent.app import generate_flutter_from_url

app = FastAPI(title="UI2Flutter Agent")

class GenerateReq(BaseModel):
    url: HttpUrl

@app.post("/generate")
async def generate(req: GenerateReq):
    out_dir = os.path.abspath("out_flutter")
    await generate_flutter_from_url(str(req.url), out_dir)
    return {"status": "ok", "out_dir": out_dir}
