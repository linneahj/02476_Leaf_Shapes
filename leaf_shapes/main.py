import json

from fastapi import FastAPI, File, UploadFile
from predict_model import LeafModel

app = FastAPI()


@app.get("/files/{file_path:path}")
async def read_file(file_path: str):
    return {"file_path": file_path}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    model = LeafModel()
    pred = model.predict_from_buffer(file.file.read())
    print(pred)
    final = json.dumps(pred)
    return final
