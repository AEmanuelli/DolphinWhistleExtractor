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


def frequency_coordinate(x):
    """A channel holding the frequency of each pixel: +1 on the top row (highest frequency), -1 on the bottom."""
    n, _, height, width = x.shape
    rows = torch.linspace(1.0, -1.0, height, device=x.device, dtype=x.dtype)
    return rows.view(1, 1, height, 1).expand(n, 1, height, width)


class TinyWhistleCNN(nn.Module):
    """
    A small CNN for embedded devices (Raspberry Pi and smaller), trained from scratch.

    It reads one channel only (the three input channels are the same grayscale spectrogram), then a
    strided 3x3 convolution and 4 stages of depthwise-separable convolutions whose width doubles from
    ``width`` to 8 x ``width``, global average pooling and a linear layer with 2 outputs.
    With width 16 it has about 40,000 parameters.

    ``coord`` adds a second input channel holding the frequency of each pixel ("CoordConv"): convolutions
    are the same at every frequency, this tells the network where in the band a pattern is.
    """

    def __init__(self, width: int = 16, dropout: float = 0.2, coord: bool = False):
        super().__init__()
        self.coord = coord

        def separable(inputs, outputs, stride):
            return [nn.Conv2d(inputs, inputs, 3, stride, 1, groups=inputs, bias=False), nn.BatchNorm2d(inputs),
                    nn.ReLU(inplace=True), nn.Conv2d(inputs, outputs, 1, bias=False), nn.BatchNorm2d(outputs),
                    nn.ReLU(inplace=True)]

        layers = [nn.Conv2d(2 if coord else 1, width, 3, 2, 1, bias=False), nn.BatchNorm2d(width),
                  nn.ReLU(inplace=True)]
        channels = width
        for stage in range(4):
            outputs = width * 2 ** stage
            layers += separable(channels, outputs, 2) + separable(outputs, outputs, 1)
            channels = outputs
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                        nn.Linear(channels, 2))

    def forward(self, x):
        x = x[:, :1]
        if self.coord:
            x = torch.cat([x, frequency_coordinate(x)], dim=1)
        return self.classifier(self.features(x))


class SubSpectralNorm(nn.Module):
    """Batch norm computed separately on ``sub_bands`` frequency sub-bands (Chang et al., 2021)."""

    def __init__(self, channels: int, sub_bands: int = 2):
        super().__init__()
        self.sub_bands = sub_bands
        self.norm = nn.BatchNorm2d(channels * sub_bands)

    def forward(self, x):
        n, c, f, t = x.shape
        if f % self.sub_bands:
            raise ValueError(f"{f} frequency rows cannot be split in {self.sub_bands} sub-bands: "
                             "use an input size that is a multiple of 16")
        return self.norm(x.reshape(n, c * self.sub_bands, f // self.sub_bands, t)).reshape(n, c, f, t)


class _BCBlock(nn.Module):
    """
    Broadcasted residual block of BC-ResNet (Kim et al., 2021, arXiv:2106.04140).

    A frequency-depthwise convolution keeps the 2-D map; its average over frequency goes through a
    temporal (dilated) depthwise convolution and a pointwise one, and is broadcast back over frequency.
    The first block of a stage ("transition") changes the number of channels and may stride in frequency.
    """

    def __init__(self, inputs, outputs, stride=1, dilation=1, dropout=0.1, sub_bands=2):
        super().__init__()
        self.transition = inputs != outputs or stride != 1
        pre = [nn.Conv2d(inputs, outputs, 1, bias=False), nn.BatchNorm2d(outputs), nn.ReLU(inplace=True)] \
            if self.transition else []
        self.f2 = nn.Sequential(*pre, nn.Conv2d(outputs, outputs, (3, 1), (stride, 1), (1, 0), groups=outputs,
                                                bias=False),
                                SubSpectralNorm(outputs, sub_bands))
        self.f1 = nn.Sequential(nn.Conv2d(outputs, outputs, (1, 3), padding=(0, dilation), dilation=(1, dilation),
                                          groups=outputs, bias=False),
                                nn.BatchNorm2d(outputs), nn.SiLU(inplace=True), nn.Conv2d(outputs, outputs, 1, bias=False),
                                nn.Dropout2d(dropout))

    def forward(self, x):
        x1 = self.f2(x)
        x2 = self.f1(x1.mean(dim=2, keepdim=True))  # (n, c, 1, t), broadcast over frequency below
        return torch.relu(x1 + x2 if self.transition else x + x1 + x2)


class BCResNet(nn.Module):
    """
    BC-ResNet for keyword spotting (Kim et al., 2021), adapted to whistle spectrograms: one input
    channel, frequency on the height (input size a multiple of 16), 2 outputs. ``tau`` scales the widths
    (BC-ResNet-1 has about 9,000 parameters). Stages of 2, 2, 4, 4 blocks with time dilations 1, 2, 4, 8;
    the frequency axis is strided by 2 in the stem and in stages 2 and 3; time is never strided.
    """

    def __init__(self, tau: float = 1.0, sub_bands: int = 2):
        super().__init__()
        base = int(8 * tau)
        widths = [2 * base, base, int(1.5 * base), 2 * base, int(2.5 * base), 4 * base]
        self.stem = nn.Sequential(nn.Conv2d(1, widths[0], 5, (2, 1), 2, bias=False), nn.BatchNorm2d(widths[0]),
                                  nn.ReLU(inplace=True))
        blocks = []
        for stage, repeats in enumerate([2, 2, 4, 4]):
            for i in range(repeats):
                inputs = widths[stage] if i == 0 else widths[stage + 1]
                stride = 2 if i == 0 and stage in (1, 2) else 1
                blocks.append(_BCBlock(inputs, widths[stage + 1], stride, 2 ** stage, sub_bands=sub_bands))
        self.blocks = nn.Sequential(*blocks)
        last = widths[-2]
        self.head = nn.Sequential(nn.Conv2d(last, last, 5, padding=(0, 2), groups=last, bias=False),
                                  nn.Conv2d(last, widths[-1], 1, bias=False), nn.BatchNorm2d(widths[-1]),
                                  nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(widths[-1], 2))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x[:, :1])))


class WhistleCRNN(nn.Module):
    """
    A small convolutional-recurrent network: 4 convolution blocks that pool the frequency axis to a few
    rows (time only twice), the average over frequency, then a bidirectional GRU along time whose outputs
    are pooled (mean and max) into 2 logits. A whistle is a contour that evolves in time: the GRU follows it.
    """

    def __init__(self, width: int = 16, hidden: int = None, dropout: float = 0.2):
        super().__init__()
        hidden = hidden or 2 * width

        def block(inputs, outputs, pool):
            return [nn.Conv2d(inputs, outputs, 3, padding=1, bias=False), nn.BatchNorm2d(outputs),
                    nn.ReLU(inplace=True), nn.MaxPool2d(pool)]

        w = width
        self.features = nn.Sequential(*block(1, w, (2, 2)), *block(w, 2 * w, (2, 2)), *block(2 * w, 4 * w, (2, 1)),
                                      *block(4 * w, 4 * w, (2, 1)))
        self.gru = nn.GRU(4 * w, hidden, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * hidden, 2))

    def forward(self, x):
        x = self.features(x[:, :1]).mean(dim=2).transpose(1, 2)  # (n, time, channels)
        x, _ = self.gru(x)
        return self.classifier(torch.cat([x.mean(dim=1), x.amax(dim=1)], dim=1))


class FrequencyDynamicConv2d(nn.Module):
    """
    Frequency dynamic convolution (Nam et al., 2022, arXiv:2203.15296): ``bases`` convolution kernels whose
    mixture depends on the frequency row. The weights come from the input averaged over time (a small 1-D
    convolution along frequency, softmax over the kernels), so the same pattern can be treated differently
    at 5 kHz and at 20 kHz, which a standard convolution cannot do.
    """

    def __init__(self, inputs, outputs, kernel_size=3, bases=4, temperature=31.0, reduction=4):
        super().__init__()
        self.bases, self.outputs, self.temperature = bases, outputs, temperature
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(torch.empty(bases * outputs, inputs, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        self.bias = nn.Parameter(torch.zeros(bases * outputs))
        hidden = max(inputs // reduction, 4)
        self.attention = nn.Sequential(nn.Conv1d(inputs, hidden, 3, padding=1, bias=False), nn.BatchNorm1d(hidden),
                                       nn.ReLU(inplace=True), nn.Conv1d(hidden, bases, 1))

    def forward(self, x):
        n, _, f, t = x.shape
        weights = torch.softmax(self.attention(x.mean(dim=3)) / self.temperature, dim=1)  # (n, bases, f)
        y = nn.functional.conv2d(x, self.weight, self.bias, padding=self.padding)  # (n, bases * outputs, f, t)
        return (y.view(n, self.bases, self.outputs, f, t) * weights[:, :, None, :, None]).sum(dim=1)


class FDYCRNN(WhistleCRNN):
    """The CRNN with frequency dynamic convolutions in blocks 2 to 4 (FDY-CRNN of Nam et al., 2022)."""

    def __init__(self, width: int = 8, hidden: int = None, dropout: float = 0.2):
        super().__init__(width, hidden, dropout)
        w = width

        def block(inputs, outputs, pool):
            return [FrequencyDynamicConv2d(inputs, outputs), nn.BatchNorm2d(outputs), nn.ReLU(inplace=True),
                    nn.MaxPool2d(pool)]

        self.features = nn.Sequential(*list(self.features)[:4], *block(w, 2 * w, (2, 2)),
                                      *block(2 * w, 4 * w, (2, 1)), *block(4 * w, 4 * w, (2, 1)))


def _tiny(width, coord=False):
    return lambda pretrained: TinyWhistleCNN(width, coord=coord)  # no ImageNet weights: trained from scratch


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
    "tinycnn_16_coord": _tiny(16, coord=True),
    "tinycnn_32_coord": _tiny(32, coord=True),
    "bcresnet_1": lambda pretrained: BCResNet(1),
    "bcresnet_2": lambda pretrained: BCResNet(2),
    "bcresnet_4": lambda pretrained: BCResNet(4),
    "bcresnet_8": lambda pretrained: BCResNet(8),
    "crnn_8": lambda pretrained: WhistleCRNN(8),
    "crnn_16": lambda pretrained: WhistleCRNN(16),
    "fdycrnn_8": lambda pretrained: FDYCRNN(8),
    "fdycrnn_16": lambda pretrained: FDYCRNN(16),
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
        elif isinstance(module, nn.Conv1d):
            macs.append(output.numel() * module.in_channels // module.groups * module.kernel_size[0])
        elif isinstance(module, FrequencyDynamicConv2d):  # the bases are all computed, then mixed
            k = module.weight.shape[-1]
            macs.append(output.numel() * module.bases * (module.weight.shape[1] * k * k + 1))
        elif isinstance(module, nn.GRU):  # 3 gates per step and direction (batch of one image)
            steps = output[0].shape[1] if module.batch_first else output[0].shape[0]
            directions = 2 if module.bidirectional else 1
            size, hidden = module.input_size, module.hidden_size
            per_step = 3 * (size * hidden + hidden * hidden)
            if module.num_layers > 1:
                per_step += (module.num_layers - 1) * 3 * (directions * hidden * hidden + hidden * hidden)
            macs.append(steps * directions * per_step)

    model = model.eval().cpu()
    hooks = [m.register_forward_hook(count) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear, nn.GRU, FrequencyDynamicConv2d))]
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
