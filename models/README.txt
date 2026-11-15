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

Older VGG models (not stored in the repository, *.h5 files are git-ignored):
  model_vgg.h5            not fine-tuned
  model_finetuned_vgg.h5  fine-tuned on our data (98.5% accuracy)
Training data: https://www.kaggle.com/datasets/alexisemanuelli/whistles-baby-hit-me-up/versions/1

/!\ Image normalisation: the fine-tuned VGG model takes normalised images (/255,
option --image_norm), whereas the initial VGG model does not. Use the setting the
model was trained with.
