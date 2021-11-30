import torch
from torchvision import transforms
from torch.nn import Sequential, Softmax

# Setup an inference pipeline with a pre-trained model
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
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
