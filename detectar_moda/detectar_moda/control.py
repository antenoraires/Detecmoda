# importar Bibliotecas
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import cv2 


# Limpesa de dados
def check_images(s_dir, ext_list):
    bad_images=[] # armaxena as imgens com errp
    bad_ext=[]  # armazena a extensão de imagens com erro
    s_list= os.listdir(s_dir) # carrega a localização das imagens
    for klass in s_list: # vai para o diretório da imagem
        klass_path=os.path.join (s_dir, klass)  
        print ('processing class directory ', klass)
        if os.path.isdir(klass_path):
            file_list=os.listdir(klass_path)
            for f in file_list:               
                f_path=os.path.join (klass_path,f)
                index=f.rfind('.')
                ext=f[index+1:].lower()
                if ext not in ext_list:  # looping para imagens com erro
                    print('file ', f_path, ' has an invalid extension ', ext)
                    bad_ext.append(f_path)
                if os.path.isfile(f_path):
                    try:
                        img = cv2.imread(f_path)
                        shape = img.shape
                        image_contents = tf.io.read_file(f_path)
                        image = tf.image.decode_jpeg(image_contents, channels=3) # abre a imagem e checa se existe ou não erro nela 
                    except Exception as e:
                        print('file ', f_path, ' is not a valid image file')
                        print(e)
                        bad_images.append(f_path)
                else:
                    print('*** fatal error, you a sub directory ', f, ' in class directory ', klass)
        else:
            print ('*** WARNING*** you have files in ', s_dir, ' it should only contain sub directories')
    return bad_images, bad_ext

def train_model(data_dir_train, img_height, img_width, batch_size):
    #para os dados de treino
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_train,
    #validation_split=0.2,
    #subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
    return train_ds

def validation_model(data_dir_validation,img_height, img_width,batch_size):
    #dados de validação
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_validation,
    #validation_split=0.2,
    #subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
    return val_ds