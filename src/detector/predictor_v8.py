import torch
import cv2
import numpy as np
import onnxruntime
from .ultils_v8 import *


class Detect_v8:
    def __init__(self, model_path=None):
        self.providers = ["CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(model_path)

    def detect(self, img):
        input = pre_process(img).detach().numpy()
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: input})[0]
        preds = non_max_suppression(torch.from_numpy(outputs))[0]
        bbox = scale_boxes([640, 640], preds[:, :4], img.shape).round()
        return preds, bbox.detach().numpy()


# if __name__ == "__main__":
#     vid = cv2.VideoCapture("3334562138064443978.mp4")
#     model = Detect_v8(model_path="weights/yolov8s_best.onnx")
#     while True:
#         ret, frame = vid.read()
#         _, bbox = model.detect(frame)
#         frame = cv2.resize(frame, (1280, 720))
#         for b in bbox:
#             b = list(map(int, b))
#             cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (255, 0, 14), 2)
#         cv2.imshow("img", frame)
#         cv2.waitKey(1)
#         # cv2.imwrite("result.jpg",img)
#         # cv2.waitKey(0)
#     cv2.destroyAllWindows()
