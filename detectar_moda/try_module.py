from detectar_moda import Pipeline
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import mlflow
import mlflow.tensorflow

# Definindo as dimensões da imagem e o tamanho do lote
img_height, img_width = 180, 180
batch_size = 32

# Inicializando o pipeline para processamento de imagens
moda = Pipeline(img_height, img_width, batch_size)

# Verificando arquivos de imagem inadequados
bad_file_list, bad_ext_list = moda.check_img()
if len(bad_file_list) != 0:
    print('Os seguintes arquivos de imagem são inadequados:')
    print(bad_file_list)
    print('Extensões inadequadas:')
    print(bad_ext_list)

# Preparando os datasets de treinamento e validação
train_ds = moda.train()
val_ds = moda.validation()

num_classes = 8  # O número de classes deve ser igual ao número de diretórios de imagens

# Inicializando o MLflow
mlflow.start_run()

# Definindo um modelo sequencial com 3 camadas
model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),  # Normalizando os pixels
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # Camada de saída para classificação
])

# Compilando o modelo com os parâmetros definidos
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Registrando parâmetros no MLflow
mlflow.log_param("img_height", img_height)
mlflow.log_param("img_width", img_width)
mlflow.log_param("batch_size", batch_size)
mlflow.log_param("num_classes", num_classes)
mlflow.log_param("epochs", 10)

# Treinando o modelo
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Registrando métricas no MLflow
mlflow.log_metric("final_accuracy", history.history['accuracy'][-1])
mlflow.log_metric("final_val_accuracy", history.history['val_accuracy'][-1])

# Salvando o modelo no MLflow
mlflow.tensorflow.log_model(model, "modelo_moda")

# Encerrando a execução do MLflow
mlflow.end_run()

# Opcional: Salvar o modelo localmente também
model.save('model/modelo_moda.h5')