import torch
from torchvision import transforms
from torch.nn import Sequential, Softmax
import torchvision.models as models
import argparse
# Setup an inference pipeline with a pre-trained model


def go(model):
    model = torch.hub.load('pytorch/vision:v0.9.0', model, pretrained=True)
    model.eval()

    pipe = Sequential(

        transforms.Resize([256, 256]),
        transforms.CenterCrop([224, 224]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        model,
        Softmax(1)
    )

    # Save inference artifact
    scripted = torch.jit.script(pipe)
    scripted.save("inference_artifact.pt")


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--model", default="resnet18",
                        help="Model to use for inference",
                        choices=["resnet18", "resnet50",
                                 "resnet101", "resnet152"]
                        )
args = arg_parser.parse_args()

if __name__ == "__main__":
    go(args.model)
