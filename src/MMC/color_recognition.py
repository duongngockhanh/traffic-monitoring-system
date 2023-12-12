import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, models, transforms
import cv2
import numpy as np
from PIL import Image

classes_name = [
    "beige",
    "black",
    "blue",
    "brown",
    "gold",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "sliver",
    "tan",
    "white",
    "yellow",
]


class Color_Recognitiion:
    def __init__(self, model_path=None):
        self.model = torchvision.models.efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(in_features=1280, out_features=15)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def preprocess(self, img):
        transform = transforms.Compose(
            [
                transforms.RandAugment(4, 4),
                transforms.Resize((132, 132)),
                transforms.ToTensor(),
            ]
        )
        return transform(img)

    def infer(self, img):
        input = self.preprocess(img)
        # print(input.shape)
        output = self.model(torch.unsqueeze(input, 0))

        _, predicted = torch.max(output, 1)
        # print(predicted)
        return classes_name[predicted[0]]


# if __name__ == "__main__":
#     img = Image.open("/Users/macos/Khanh_hoa_project/test_model.jpeg")
#     model = Color_Recognitiion(
#         "/Users/macos/Khanh_hoa_project/weights/best_weights.pth"
#     )
#     output = model.infer(img)
#     print(output)
