from ultralytics import YOLO
import cv2
import torch

path_weight = "yolov8n.pt"

model = YOLO(path_weight)


def infer_image(img_path):
    img = cv2.imread(img_path)
    predictions = []
    for r in model.predict(img):
        for box in r.boxes:
            list_box = torch.tensor(box.xywhn[0].tolist())
            prediction = torch.cat((box.cls, list_box))
            predictions.append(prediction)

    predictions = torch.vstack(predictions)
    return predictions  # (class, x, y, w, h)


# if __name__ == "__main__":
#     img_path = "abc.jpg" # default is bus.jpg
#     print(infer_image(img_path))
