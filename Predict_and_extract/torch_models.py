"""
PyTorch whistle classifiers shared by the detection pipeline and the training script.

Every model takes 224x224 RGB images (the grayscale spectrogram replicated on 3 channels),
scaled to 0-1 and normalised with the ImageNet statistics, and returns 2 logits (noise, whistle).

A checkpoint written by ``Train_model/train_openwhistle.py`` stores the architecture name
under ``"arch"``; checkpoints without it are the VGG16 of the OpenWhistle paper.
"""
import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PAPER_VGG = "vgg16_openwhistle"


class VGG16WhistleClassifier(nn.Module):
    """
    The detector of the OpenWhistle paper (cnn/utils/model.py of the official code):
    VGG16 convolutional backbone, fully connected layers of 50 and 20 units, 2 outputs.
    Attribute names match the published checkpoint (model_vgg_final_best.pt).
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()
        from torchvision import models

        backbone = models.vgg16(weights="IMAGENET1K_V1" if pretrained else None)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(backbone.classifier[0].in_features, 50)
        self.fc2 = nn.Linear(50, 20)
        self.out = nn.Linear(20, 2)

    def forward(self, x):
        x = self.flatten(self.avgpool(self.features(x)))
        return self.out(torch.relu(self.fc2(torch.relu(self.fc1(x)))))


def _replace_last_linear(sequential: nn.Sequential, outputs: int = 2):
    sequential[-1] = nn.Linear(sequential[-1].in_features, outputs)


def _torchvision(name, head):
    def build(pretrained):
        from torchvision import models

        model = getattr(models, name)(weights="DEFAULT" if pretrained else None)
        head(model)
        return model
    return build


ARCHITECTURES = {
    PAPER_VGG: VGG16WhistleClassifier,
    "mobilenet_v2": _torchvision("mobilenet_v2", lambda m: _replace_last_linear(m.classifier)),
    "mobilenet_v3_large": _torchvision("mobilenet_v3_large", lambda m: _replace_last_linear(m.classifier)),
    "efficientnet_b0": _torchvision("efficientnet_b0", lambda m: _replace_last_linear(m.classifier)),
    "resnet18": _torchvision("resnet18", lambda m: setattr(m, "fc", nn.Linear(m.fc.in_features, 2))),
    "resnet50": _torchvision("resnet50", lambda m: setattr(m, "fc", nn.Linear(m.fc.in_features, 2))),
    "convnext_tiny": _torchvision("convnext_tiny", lambda m: _replace_last_linear(m.classifier)),
}


def build_model(arch: str, pretrained: bool = False) -> nn.Module:
    """A 2-class whistle classifier; ``pretrained`` starts from the ImageNet weights (downloaded once)."""
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {arch!r}, expected one of {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[arch](pretrained)


def load_checkpoint(path: str):
    """
    Read a checkpoint without executing code (``weights_only``).

    Returns (model in eval mode on CPU, metadata dict). Accepts the published checkpoint of the
    paper, checkpoints of train_openwhistle.py and bare state dicts of the paper's VGG16.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else {}
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model = build_model(metadata.get("arch") or PAPER_VGG)
    model.load_state_dict(state_dict)
    return model.eval(), metadata
