import torch
from torchvision import transforms
from PIL import Image
import numpy as np


pipe_reload = torch.jit.load("inference_artifact.pt")


classes = []
with open("imagenet_classes.txt", "r") as f:
    classes = [s.strip() for s in f.readlines()]


def predict(file):
    img = Image.open(file)

    img.load()
# Make into a batch of 1 element
    data = transforms.ToTensor()(np.asarray(img, dtype="uint8").copy()).unsqueeze(0)

# Perform inference
    with torch.no_grad():
        logits = pipe_reload(data).detach()

    proba = logits[0]

# Transform to class and print answer

    return classes[proba.argmax()]
