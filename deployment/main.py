import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from PIL import Image, ImageEnhance

from keras.layers import *
from keras.models import *
from keras.metrics import *
from keras.optimizers import *
from keras.applications import *
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
import seaborn as sns
import random
from flask import Flask, render_template, request
import base64
import os



trainDataset = ['deployment/DATA/Training/']
testDataset = ['deployment/DATA/Testing/']

trainPaths = []
trainLables = []

for trainDir in trainDataset:
    for label in os.listdir(trainDir):
        for image in os.listdir(trainDir+label):
            trainPaths.append(trainDir+label+'/'+image)
            trainLables.append(label)

trainPaths, trainLables = shuffle(trainPaths, trainLables)

testPaths = []
testLables = []

for testDir in testDataset:
    for label in os.listdir(testDir):
        for image in os.listdir(testDir+label):
            testPaths.append(testDir+label+'/'+image)
            testLables.append(label)

testPaths, testLables = shuffle(testPaths, testLables)
uniqueLables = os.listdir(trainDir)


agu = 0.5
def augmentImages(image):
    image = Image.fromarray(np.uint8(image))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(1-agu,1+agu))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(1-agu,1+agu))
    image = ImageEnhance.Sharpness(image).enhance(random.uniform(1-agu,1+agu))
    
    image = np.array(image)/255.0
    return image
    
imageSize = 128


        # FOR VISULISIN THE SET
def openImages(paths):
    images = []
    for path in paths:
        image = load_img(path, target_size=(imageSize,imageSize))
        image = augmentImages(image)
        images.append(image)
    return np.array(images)

def encodeLables(labels):
    encoded = []
    for x in labels:
        encoded.append(uniqueLables.index(x))
    return np.array(encoded)



try:
    model = tf.keras.models.load_model('../dev/modeltst.keras')
    print("LOADED MODEL")
except Exception as e:
    print(e)
    def datagen(paths, labels, batchSize=12, epochs=1):
        for _ in range(epochs):
            for x in range(0, len(paths), batchSize):
                batchPaths = paths[x:x+batchSize]
                batchImages = openImages(batchPaths)
                batchLabels = labels[x:x+batchSize]
                batchLabels = encodeLables(batchLabels)
                yield batchImages, batchLabels

    baseModel = VGG16(input_shape=(imageSize,imageSize,3), include_top=False, weights='imagenet')
    for layer in baseModel.layers:
        layer.trainable = False
    # Set the last vgg block to trainable
    baseModel.layers[-2].trainable = True
    baseModel.layers[-3].trainable = True
    baseModel.layers[-4].trainable = True

    model = Sequential()
    model.add(Input(shape=(imageSize,imageSize,3)))
    model.add(baseModel)
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(len(uniqueLables), activation='softmax'))


    model.compile(optimizer=Adam(learning_rate=0.0001),
             loss='sparse_categorical_crossentropy',
             metrics=['sparse_categorical_accuracy'])
    
    batchSize = 25
    steps = int(len(trainPaths)/batchSize)
    epochs = 1
    history = model.fit(datagen(trainPaths, trainLables, batchSize=batchSize, epochs=epochs),
                        epochs=epochs, steps_per_epoch=steps)
    model.save("model.keras")
    print("YOU NEED TO RENAME MODEL2.KERAS TO MODEL.KERAS")


###################################################
            # AFTER ALL THAT #
###################################################





# to see the confusion matrix run the code below
            
# minSamplesPerClass = min([testLables.count(label) for label in uniqueLables])
# balancedTestPaths = []
# balancedTestLables = []

# for label in uniqueLables:
#     classPaths = [path for path, lbl in zip(testPaths, testLables) if lbl == label]
#     balancedTestPaths.extend(classPaths[:minSamplesPerClass])
#     balancedTestLables.extend([label] * minSamplesPerClass)


# testImagesBalanced = openImages(balancedTestPaths)
# testLablesEncodedBalanced = encodeLables(balancedTestLables)

# predictionsBalanced = model.predict(testImagesBalanced)
# predictedLabelsBalanced = np.argmax(predictionsBalanced, axis=1)

# confusionMatrixBalanced = tf.math.confusion_matrix(testLablesEncodedBalanced, predictedLabelsBalanced, num_classes=len(uniqueLables))

# plt.figure(figsize=(8, 6))
# sns.heatmap(confusionMatrixBalanced, annot=True, fmt='d', cmap='Blues', xticklabels=uniqueLables, yticklabels=uniqueLables)
# plt.title('Confusion Matrix (Balanced Testing)')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()
# accuracy_balanced = np.sum(np.diag(confusionMatrixBalanced)) / np.sum(confusionMatrixBalanced)
# print(f"Accuracy on Balanced Testing (100 samples): {accuracy_balanced:.2%}")


###################################
            #prediction
###################################
def DECLB(label): # decode lables
    return uniqueLables[label]

def prd(path):
    image = load_img(path, target_size=(imageSize,imageSize))
    image = augmentImages(image)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    print(f"prediction: {prediction}")
    predictedLabel = DECLB(np.argmax(prediction))
    print(f"predictiedLable: {predictedLabel}")
    certainty = round(np.max(prediction) * 100)
    return predictedLabel, certainty


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
            prdLabel, cert = prd(loco)
            if cert >= 90:
                cert = f"high {cert}% certainty"
            elif cert >= 80:
                cert = f"{cert}% certainty"
            elif cert >=65:
                cert = f"low certainty of {cert}%"
            else:
                cert = f"very low certainty, only {cert}%"
            with open(loco, "rb") as image_file:
                encodedString = base64.b64encode(image_file.read()).decode('utf-8') # turn file into base63
            return render_template('results.html', imageData=encodedString, label=prdLabel, certainty=cert)

        else:
            return 'No file uploaded'
        
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=80)
