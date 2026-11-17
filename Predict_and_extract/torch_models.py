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


class TinyWhistleCNN(nn.Module):
    """
    A small CNN for embedded devices (Raspberry Pi and smaller), trained from scratch.

    It reads one channel only (the three input channels are the same grayscale spectrogram), then a
    strided 3x3 convolution and 4 stages of depthwise-separable convolutions whose width doubles from
    ``width`` to 8 x ``width``, global average pooling and a linear layer with 2 outputs.
    With width 16 it has about 40,000 parameters.
    """

    def __init__(self, width: int = 16, dropout: float = 0.2):
        super().__init__()

        def separable(inputs, outputs, stride):
            return [nn.Conv2d(inputs, inputs, 3, stride, 1, groups=inputs, bias=False), nn.BatchNorm2d(inputs),
                    nn.ReLU(inplace=True), nn.Conv2d(inputs, outputs, 1, bias=False), nn.BatchNorm2d(outputs),
                    nn.ReLU(inplace=True)]

        layers = [nn.Conv2d(1, width, 3, 2, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True)]
        channels = width
        for stage in range(4):
            outputs = width * 2 ** stage
            layers += separable(channels, outputs, 2) + separable(outputs, outputs, 1)
            channels = outputs
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                        nn.Linear(channels, 2))

    def forward(self, x):
        return self.classifier(self.features(x[:, :1]))


def _tiny(width):
    return lambda pretrained: TinyWhistleCNN(width)  # no ImageNet weights: always trained from scratch


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
    "mobilenet_v3_small": _torchvision("mobilenet_v3_small", lambda m: _replace_last_linear(m.classifier)),
    "tinycnn_8": _tiny(8),
    "tinycnn_16": _tiny(16),
    "tinycnn_32": _tiny(32),
    "tinycnn_64": _tiny(64),
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


def model_cost(model: nn.Module, size: int = 224, threads: int = 1, repeats: int = 30):
    """
    Size and speed of a model: parameters, multiply-accumulates of one image (convolutions and linear
    layers), and the median CPU latency of one image with ``threads`` threads (a rough proxy of an
    embedded device: a Raspberry Pi 4 core is a few times slower than a laptop core).
    """
    import statistics
    import time

    macs = []

    def count(module, inputs, output):
        if isinstance(module, nn.Conv2d):
            macs.append(output.numel() * module.in_channels // module.groups * module.kernel_size[0]
                        * module.kernel_size[1])
        elif isinstance(module, nn.Linear):
            macs.append(module.in_features * module.out_features)

    model = model.eval().cpu()
    hooks = [m.register_forward_hook(count) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    x = torch.zeros(1, 3, size, size)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        with torch.inference_mode():
            model(x)
            for hook in hooks:
                hook.remove()
            times = []
            for _ in range(repeats):
                started = time.perf_counter()
                model(x)
                times.append(time.perf_counter() - started)
    finally:
        torch.set_num_threads(previous_threads)
    return {"params": sum(p.numel() for p in model.parameters()), "macs": int(sum(macs)),
            "cpu_ms": 1000 * statistics.median(times)}


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
