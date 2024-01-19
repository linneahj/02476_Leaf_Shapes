import json
import os

from fastapi import FastAPI, File, UploadFile

from leaf_shapes.predict_model import LeafModel

app = FastAPI()

model_path = os.getenv("MODEL_PATH", "./models/model_best.pth.tar")
class_ids_path = os.getenv("CLASS_IDS_PATH", "./data/processed/Class_ids.csv")
print(model_path)
print(class_ids_path)

model = LeafModel(model_path, class_ids_path)


@app.get("/files/{file_path:path}")
async def read_file(file_path: str):
    return {"file_path": file_path}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    pred = model.predict_from_buffer(file.file.read())
    print(pred)
    final = json.dumps(pred)
    return final
