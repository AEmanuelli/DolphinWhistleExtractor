Pre-trained models used by the detection pipeline.

  base_MobileNetV2.keras                default model (used by main.py and the GUI)
  base_mobilenetv2_best_val_loss.keras  MobileNetV2 checkpoint with the best validation loss
  base_EfficientNetB0.keras             EfficientNetB0 variant

All three take 224x224x3 spectrogram images with RAW pixel values (0-255) and
output a single sigmoid score (probability that the window contains a whistle).
Do NOT use --image_norm with them: on images divided by 255 they predict
"noise" for every window.

Accuracy on the OpenWhistle-CNN test split (16,708 windows, threshold 0.5), see
the "Model evaluation" section of the main README and Train_model/evaluate_models.py:
  base_MobileNetV2                97.9%
  base_EfficientNetB0             95.0%
  base_mobilenetv2_best_val_loss  94.5%

The VGG16 detector of the OpenWhistle paper (arXiv:2609.34839, 97.2% on the same
test split) is not stored here. Download its PyTorch checkpoint and pass it to
--model_path (requires torch and torchvision):
  https://huggingface.co/dolphinteam/OpenWhistle-CNN-VGG16/resolve/main/model_vgg_final_best.pt

Older Keras VGG models (not stored in the repository, *.h5 files are git-ignored).
They follow Korkmaz et al. 2023 (doi:10.3389/frai.2023.1099022), whose images are
0.8 s windows in the 3-20 kHz band: run them with --spectrogram legacy.
  model_vgg.h5            not fine-tuned
  model_finetuned_vgg.h5  fine-tuned on our data (98.5% accuracy)
Training data: https://www.kaggle.com/datasets/alexisemanuelli/whistles-baby-hit-me-up/versions/1

/!\ Image normalisation: the fine-tuned VGG model takes normalised images (/255,
option --image_norm), whereas the initial VGG model does not. Use the setting the
model was trained with.
