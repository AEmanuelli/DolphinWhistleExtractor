# %% [code]
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import os

def send_email(username=None, password=None, to_address=None, subject='Sujet de l\'email', msg='Contenu de l\'email', file_path=None):
    """Send an e-mail through Gmail.

    Credentials are never stored in the code: they are read from the
    EMAIL_USERNAME / EMAIL_APP_PASSWORD environment variables (e.g. Kaggle
    secrets) when not given explicitly. EMAIL_TO defaults to the sender.
    """
    username = username or os.environ.get("EMAIL_USERNAME")
    password = password or os.environ.get("EMAIL_APP_PASSWORD")
    to_address = to_address or os.environ.get("EMAIL_TO") or username
    if not username or not password:
        raise RuntimeError("Set EMAIL_USERNAME and EMAIL_APP_PASSWORD to send e-mails.")

    # Création de l'objet MIMEMultipart
    message = MIMEMultipart()

    # Ajout du sujet et du contenu au MIMEMultipart
    message['Subject'] = subject
    message.attach(MIMEText(msg))

    # Vérification si file_path est spécifié
    if file_path is not None:
        with open(file_path, "rb") as attachment:
            attachment_part = MIMEApplication(attachment.read(), Name=os.path.basename(file_path))
            attachment_part["Content-Disposition"] = f"attachment; filename={os.path.basename(file_path)}"
            message.attach(attachment_part)

    # Connexion au serveur SMTP de Gmail
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(username, password)

    # Envoi de l'email
    server.sendmail(username, to_address, message.as_string())
    server.quit()

    print('Email envoyé avec succès !')
    
    #********************* IMPORTS

import tensorflow as tf 
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from os import listdir
from os.path import isfile, join
import cv2
import shutil
import pandas as pd


#%%

#********************* FUNCTIONS

def predict(model, images_path, file_path):
    image = cv2.imread(join(images_path, file_path))
    image = cv2.resize(image, (224, 224))
    image = np.resize(image,(1,224,224,3))
    image = preprocess_input(image)
    prediction = model.predict(image, verbose = 0)
    
    return prediction

def move_file(images_path, file_path, dst_dir):
    src_dir = join(images_path, file_path)
    shutil.move(src_dir,dst_dir)

def copy_file(images_path, file_path, dst_dir):
    src_dir = join(images_path, file_path)
    shutil.copy(src_dir,dst_dir)
    
def prepare_csv_data(file_path, record_names, positive_initial, positive_finish):
    part = file_path.split('-')

    name = part[0:-1]
    record_names.append(name)
    
    ini = part[-1].replace(".jpg", "")
    ini = float(ini)
    positive_initial.append(ini)
    
    fin = ini + 0.8
    fin = round(fin, 1)
    positive_finish.append(fin)
    
    return record_names, positive_initial, positive_finish

def save_csv(record_names, positive_initial, positive_finish, class_1_scores, csv_path):
    df = {'file_name': record_names,
    'initial_point': positive_initial,
    'finish_point': positive_finish,
    'confidence': class_1_scores}

    df = pd.DataFrame(df)
    
    df.to_csv(csv_path, index=False)

#%%
