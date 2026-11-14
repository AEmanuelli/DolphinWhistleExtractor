Pre-trained models used by the detection pipeline.

  base_MobileNetV2.keras                default model (used by main.py and the GUI)
  base_mobilenetv2_best_val_loss.keras  MobileNetV2 checkpoint with the best validation loss
  base_EfficientNetB0.keras             EfficientNetB0 variant

All three take 224x224x3 spectrogram images and output a single sigmoid score
(probability that the window contains a whistle).

Older VGG models (not stored in the repository, *.h5 files are git-ignored):
  model_vgg.h5            not fine-tuned
  model_finetuned_vgg.h5  fine-tuned on our data (98.5% accuracy)
Training data: https://www.kaggle.com/datasets/alexisemanuelli/whistles-baby-hit-me-up/versions/1

/!\ Image normalisation: the fine-tuned VGG model takes normalised images (/255,
option --image_norm), whereas the initial VGG model does not. Use the setting the
model was trained with.
