import torch

def tta_predict(model, images):
    outputs1 = model(images)

    flipped = torch.flip(images, dims=[3])
    outputs2 = model(flipped)
    outputs2 = torch.flip(outputs2, dims=[3])

    return (outputs1 + outputs2) / 2