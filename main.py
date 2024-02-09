import tensorflow as tf
print("TensorFlow version:", tf.__version__)
mnist = tf.keras.datasets.fashion_mnist

dta = {0:"T-shirt/top", 1:"Trouser", 2:"Pullover", 3:"Dress", 4:"Coat", 5:"Sandal", 6:"Shirt", 7:"Sneaker", 8:"Bag", 9:"Ankle boot"}

dataTestDir = "./DATA/Testing"
dataTrainDir = "./DATA/Training"
imgSize = (512, 512)
batchSize = 32

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataTrainDir,
    seed = 1000,
    shuffle = True,
    image_size = imgSize,
    batch_size = batchSize
)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

import numpy as np
import pandas as pd
# np.shape(mnist.load_data())
import matplotlib.pyplot as plt
# image = x_train[5]
# print(y_train[5])

# plot the sample
# fig = plt.figure
# plt.imshow(image)
# plt.show()


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# predictions = model(x_train[:1]).numpy()



# tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=1)

model.evaluate(x_test,  y_test, verbose=2)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

val = 0
num_images = 20
out = probability_model(x_test[val:val + num_images])
predicted_classes = [np.argmax(i) for i in out]

num_cols = min(num_images, 5)
num_rows = -(-num_images // num_cols)  

plt.figure(figsize=(15, 3 * num_rows))

for i in range(num_images):
    plt.subplot(num_rows, num_cols, i + 1)
    plt.imshow(x_test[val + i])
    plt.title(f"{dta[predicted_classes[i]]} {'True' if predicted_classes[i] == y_test[val + i] else 'FALSE'}")
    plt.axis('off')

plt.show()