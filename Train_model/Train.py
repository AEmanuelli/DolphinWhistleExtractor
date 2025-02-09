import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve, auc
from scipy import interp
import numpy as np
import gc

#%% NECESSARY VARIABLES

IMG_SIZE = 224
batch_size = 32
directory = "DNN_whistle_detection/Train_model"

train_data = pd.read_csv('DNN_whistle_detection/Train_model/dolphin_signal_train.csv')
train_data = train_data.sample(frac=1, random_state=2).reset_index() # shuffle data


train_data = train_data.astype({"labels" : str})

labels = train_data['labels']

skf = StratifiedKFold(n_splits = 5, random_state = 7, shuffle = True)

idg = ImageDataGenerator(preprocessing_function = preprocess_input) 

#%% MODEL

def model(IMG_SIZE):
    ## Loading VGG16 model
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
    base_model.trainable = True 
    
    model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(50, activation='relu'),
    layers.Dense(20, activation='relu'),
    layers.Dense(2, activation='softmax')])
    return model

#%%
fpr_plot_list = []
tpr_plot_list = []
label_plot_list = []

first_run = True

list_loss = []
list_acc = []

tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)

counter = 1
# enumerate splits
for train, val in skf.split(train_data, labels):
    print('train: %s, test: %s' % (train_data.iloc[train], train_data.iloc[val]))
    
    training_data = train_data.iloc[train]
    validation_data = train_data.iloc[val]

    del train, val
    gc.collect()
    
    train_data_generator = idg.flow_from_dataframe(dataframe = training_data,
                                                   directory = directory,
                                                   x_col= 'file_names',
                                                   y_col= 'labels',
                                                   batch_size = batch_size,
                                                   seed = None,
                                                   shuffle = False,
                                                   class_mode='categorical',
                                                   classes = ["0","1"],
                                                   color_mode="rgb",
                                                   target_size = (IMG_SIZE, IMG_SIZE))

    valid_data_generator  = idg.flow_from_dataframe(dataframe = validation_data,
                                                   directory = directory,
                                                   x_col= 'file_names',
                                                   y_col= 'labels',
                                                   batch_size = batch_size,
                                                   seed = None,
                                                   shuffle = False,
                                                   class_mode='categorical',
                                                   classes = ["0","1"],
                                                   color_mode="rgb",
                                                   target_size = (IMG_SIZE, IMG_SIZE))
    
   
    if first_run:
        model = model(IMG_SIZE)
        first_run = False
        
    else:
        model = load_model("models\\model_vgg.h5")
    
    
    optimizer = Adam(learning_rate = 0.00001, beta_1=0.9, beta_2=0.999,epsilon=1e-7,amsgrad=False,name='Adam')    
    model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', mode='min', patience=15,  restore_best_weights=True)
    
    history = model.fit(
        train_data_generator,
        epochs=100,
        validation_data=valid_data_generator,
        batch_size=batch_size, 
        callbacks=[es])  
    
    # summarize history for loss
    plt.plot(model.history.history['loss'])
    plt.plot(model.history.history['val_loss'])
    title = "model loss " + str(counter)
    plt.title(title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt_title = "figs\\" + title + ".fold.png"
    plt.savefig(plt_title)
    plt.show()   
                  
    # summarize history for acc
    plt.plot(model.history.history['accuracy'])
    plt.plot(model.history.history['val_accuracy'])
    title = "model accuracy " + str(counter)
    plt.title(title)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt_title = "figs\\" + title + ".fold.png"
    plt.savefig(plt_title)
    plt.show()   
    

    loss, acc = model.evaluate(valid_data_generator, verbose=2)
    list_loss.append(loss)
    list_acc.append(acc)
    
    
    # for avg roc
    predictions = model.predict_generator(valid_data_generator, steps = len(valid_data_generator.labels))
    predictions = np.array(predictions)
    fpr, tpr, _ = roc_curve(validation_data["labels"].astype(int), predictions[:, 1], pos_label = 1)
    interp_tpr = interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0
    tprs.append(interp_tpr)
    roc_auc = auc(fpr, tpr)
    
    fpr_plot_list.append(fpr)
    tpr_plot_list.append(tpr)
    label_plot_list.append(str('ROC fold %d (AUC = %0.2f)' % (counter, roc_auc)))
          
    model.save("models\\model_vgg.h5")
    
    counter+=1
    del training_data, validation_data
    gc.collect()

#%%  

avg_loss =  sum(list_loss) / len(list_loss)
avg_acc = sum(list_acc) / len(list_acc)
print("avg loss:", str(avg_loss), "avg acc:", avg_acc)

#%%
for i in range(len(tpr_plot_list)):
    plt.plot(fpr_plot_list[i], tpr_plot_list[i], lw=2, alpha=0.3, label=label_plot_list[i])

plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")

title = "figs\\avg_roc_vgg.png"
plt.savefig(title)
plt.show()

#%%