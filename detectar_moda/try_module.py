from  detectar_moda import Pipeline


img_height,img_width= 180, 180 #Definindo as dimensões da imagem.
batch_size=32

moda = Pipeline(img_height, img_width, batch_size)

train_ds = moda.train()

class_names = train_ds.class_names
print(class_names)