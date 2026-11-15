Two windows of the test split of the OpenWhistle-CNN dataset
(https://huggingface.co/datasets/dolphinteam/OpenWhistle-CNN, CC BY 4.0;
Mustun et al., "OpenWhistle: A Large-Scale Longitudinal Dataset and Benchmark of
Bottlenose Dolphin Vocalizations", arXiv:2609.34839), unmodified:

- `whistle.wav` / `whistle.png`: `manual_SW_Nana_13-11-19-1221_001217330_001217730.wav` (label 1)
- `noise.wav` / `noise.png`: `noise_13-11-19-1221_000245600_000246000.wav` (label 0)

The `.wav` files are the 0.4 s audio windows and the `.png` files the spectrogram
images shipped with the dataset. The tests use them to check that this repository
computes the same spectrograms as the paper.
