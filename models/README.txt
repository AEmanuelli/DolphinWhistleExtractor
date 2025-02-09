This is where you should store the models. 



the model non finetuned is called model_vgg.h5
the other one is the model finetuned on our data, with 98,5% accuracy.
this is the data "https://www.kaggle.com/datasets/alexisemanuelli/whistles-baby-hit-me-up/versions/1"

BUT the test dataset used to verify accuracy is stored here : /media/DOLPHIN_ALEXIS/Analyses_alexis/Spectrograms_datasets/dataset/_last/tests/test

/!/ The model finetuned takes normalised images as inputs (/255) whereas the initial one does not.
