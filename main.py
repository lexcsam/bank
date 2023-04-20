from typing import Union
from prediction import prediction
from fastapi import FastAPI,Request
import numpy as np

app = FastAPI()


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


