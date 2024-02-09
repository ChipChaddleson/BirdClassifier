import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image, ImageEnhance

from tensorflow import keras
from keras.layers import *
from keras.models import *
from keras.metrics import *
from keras.optimizers import *
from keras.applications import *
from keras.preprocessing.image import load_img

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
import os
import random

from flask import Flask, render_template, request
import os



train_dataset = ['deployment/DATA/Training/']
test_dataset = ['deployment/DATA/Testing/']

train_paths = []
train_labels = []

for train_dir in train_dataset:
    for label in os.listdir(train_dir):
        for image in os.listdir(train_dir+label):
            train_paths.append(train_dir+label+'/'+image)
            train_labels.append(label)

train_paths, train_labels = shuffle(train_paths, train_labels)

test_paths = []
test_labels = []

for test_dir in test_dataset:
    for label in os.listdir(test_dir):
        for image in os.listdir(test_dir+label):
            test_paths.append(test_dir+label+'/'+image)
            test_labels.append(label)

test_paths, test_labels = shuffle(test_paths, test_labels)
unique_labels = os.listdir(train_dir)


agu = 0.5
def augment_image(image):
    image = Image.fromarray(np.uint8(image))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(1-agu,1+agu))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(1-agu,1+agu))
    image = ImageEnhance.Sharpness(image).enhance(random.uniform(1-agu,1+agu))
    
    image = np.array(image)/255.0
    return image
    
IMAGE_SIZE = 128


        # FOR VISULISIN THE SET
def open_images(paths):
    '''
    Given a list of paths to images, this function returns the images as arrays (after augmenting them)
    '''
    images = []
    for path in paths:
        image = load_img(path, target_size=(IMAGE_SIZE,IMAGE_SIZE))
        image = augment_image(image)
        images.append(image)
    return np.array(images)

def encode_label(labels):
    encoded = []
    for x in labels:
        encoded.append(unique_labels.index(x))
    return np.array(encoded)

def decode_label(labels):
    decoded = []
    for x in labels:
        decoded.append(unique_labels[x])
    return np.array(decoded)
try:
    model = tf.keras.models.load_model('deployment/model.keras')
    print("LOADED MODEL")
except Exception as e:
    print(e)

    # images = open_images(train_paths[40:49])
    # lbl = train_labels[40:49]
    # fig = plt.figure(figsize=(12, 6))
    # for x in range(1, 9):
    #     fig.add_subplot(2, 4, x)
    #     plt.axis('off')
    #     plt.title(lbl[x])
    #     plt.imshow(images[x])
    # plt.rcParams.update({'font.size': 12})
    # plt.show()


    def datagen(paths, labels, batch_size=12, epochs=1):
        for _ in range(epochs):
            for x in range(0, len(paths), batch_size):
                batch_paths = paths[x:x+batch_size]
                batch_images = open_images(batch_paths)
                batch_labels = labels[x:x+batch_size]
                batch_labels = encode_label(batch_labels)
                yield batch_images, batch_labels

    base_model = VGG16(input_shape=(IMAGE_SIZE,IMAGE_SIZE,3), include_top=False, weights='imagenet')
    for layer in base_model.layers:
        layer.trainable = False
    # Set the last vgg block to trainable
    base_model.layers[-2].trainable = True
    base_model.layers[-3].trainable = True
    base_model.layers[-4].trainable = True

    model = Sequential()
    model.add(Input(shape=(IMAGE_SIZE,IMAGE_SIZE,3)))
    model.add(base_model)
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(len(unique_labels), activation='softmax'))


    model.compile(optimizer=Adam(learning_rate=0.0001),
             loss='sparse_categorical_crossentropy',
             metrics=['sparse_categorical_accuracy'])
    
    batch_size = 25
    steps = int(len(train_paths)/batch_size)
    epochs = 1
    history = model.fit(datagen(train_paths, train_labels, batch_size=batch_size, epochs=epochs),
                        epochs=epochs, steps_per_epoch=steps)
    model.save("model2.keras")
    print("YOU NEED TO RENAME MODEL2.KERAS TO MODEL.KERAS")

    batch_size = 32
    steps = int(len(test_paths)/batch_size)
    y_pred = []
    y_true = []
    for x,y in tqdm(datagen(test_paths, test_labels, batch_size=batch_size, epochs=1), total=steps):
        pred = model.predict(x)
        pred = np.argmax(pred, axis=-1)
        for i in decode_label(pred):
            y_pred.append(i)
        for i in decode_label(y):
            y_true.append(i)


###################################################
            # AFTER ALL THAT #
###################################################





# to see the confusion matrix run the code below
            
# min_samples_per_class = min([test_labels.count(label) for label in unique_labels])
# balanced_test_paths = []
# balanced_test_labels = []

# for label in unique_labels:
#     class_paths = [path for path, lbl in zip(test_paths, test_labels) if lbl == label]
#     balanced_test_paths.extend(class_paths[:min_samples_per_class])
#     balanced_test_labels.extend([label] * min_samples_per_class)


# test_images_balanced = open_images(balanced_test_paths)
# test_labels_encoded_balanced = encode_label(balanced_test_labels)

# predictions_balanced = model.predict(test_images_balanced)
# predicted_labels_balanced = np.argmax(predictions_balanced, axis=1)

# confusion_matrix_balanced = tf.math.confusion_matrix(test_labels_encoded_balanced, predicted_labels_balanced, num_classes=len(unique_labels))

# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix_balanced, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
# plt.title('Confusion Matrix (Balanced Testing)')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()


###################################
            #prediction
###################################
def DECLB(label): # decode lables
    return unique_labels[label]

def prd(path):
    image = load_img(path, target_size=(IMAGE_SIZE,IMAGE_SIZE))
    image = augment_image(image)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    predicted_label = DECLB(np.argmax(prediction))
    certainty = round(np.max(prediction) * 100)
    return predicted_label, certainty


###################################
        # web server #
###################################


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def uploadFile():
    if request.method == 'POST':
        uploadedFile = request.files['file']
        if uploadedFile:
            tempDir = os.path.join(os.path.dirname(__file__), 'temp')
            if not os.path.exists(tempDir):
                os.makedirs(tempDir)
            loco = os.path.join(tempDir, uploadedFile.filename)
            uploadedFile.save(loco)
            prdLabel, Cert = prd(loco)
            return f'this is a {prdLabel}, with {Cert}% chance'
        else:
            return 'No file uploaded'

if __name__ == '__main__':
    app.run(debug=True)

