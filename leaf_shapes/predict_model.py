import cv2
import numpy as np
import pandas as pd
import timm
import torch


class LeafModel:
    def __init__(self, model_path, class_ids_path) -> None:
        # model_path = "../models/model_best.pth.tar"  # Model to be used must be placed here after training finishes
        self.model = timm.create_model("resnet18", pretrained=True, checkpoint_path=model_path, num_classes=99)
        self.labels = pd.read_csv(class_ids_path)["0"].tolist()
        print(self.model.num_classes)
        print(len(self.labels))

    def eval(self):
        self.model.eval()

    def predict_from_buffer(self, img_buf):
        img_np = np.asarray(bytearray(img_buf), dtype="uint8")
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        return self.predict_from_img(img)

    def predict(self, img_path):
        img = cv2.imread(img_path)
        return self.predict_from_img(img)

    def predict_from_img(self, img):
        self.model.eval()
        with torch.no_grad():
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(np.moveaxis(img, 2, 0))

            output = self.model(t.float().unsqueeze(0))
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

            values, indices = torch.topk(probabilities, 5)

            result = [self.labels[i] for i in indices]
            probs = [values[i].item() * 100 for i in range(5)]

            final_result = [{"prediction": result[i], "probability": probs[i]} for i in range(5)]

            return final_result


if __name__ == "__main__":
    model_path = "../models/model_best.pth.tar"
    class_ids_path = "../data/processed/Class_ids.csv"
    model = LeafModel(model_path, class_ids_path)

    pred = model.predict("../data/processed/TIMM/train/Acer_Capillipes/610.png")
    print(pred)
