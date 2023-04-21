from typing import Union
from prediction import prediction
from fastapi import FastAPI,Request
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
    "https://example.com",
    "https://example.com:8443",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/")
async def read_root(request: Request):
   result=0
   try:
        data=await request.json()
        res=await prediction(data)
        result=np.int16(res).item()
        return {"result": result}
   except:
        return {"result": -1}


