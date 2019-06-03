import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import reuters
from keras.datasets import boston_housing
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
from keras.layers import Dropout

# we have 86% of accuracy, best I found have 94% https://www.kaggle.com/nilanml/imdb-review-deep-model-94-89-accuracy

print (keras.__version__)

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000) # reuters.load_data(num_words=10000)
num_classes=np.max(train_labels) + 1
print ("Number of classes: ", num_classes)

print ('Example data point from index 10:', train_data[10])

word_index=imdb.get_word_index()  # reuters.get_word_idex
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

decoded_item = ' '.join([reverse_word_index.get(i-3,'?') for i in train_data[10]])
print ('Decoded item: ', decoded_item)

tokenizer = Tokenizer(num_words=10000)
train_data = tokenizer.sequences_to_matrix(train_data, mode='binary')
test_data = tokenizer.sequences_to_matrix(test_data, mode='binary')

print('Train seqence length :', len(train_data))
print('Test seqence length :', len(test_data))

one_hot_train_labels = to_categorical(train_labels, num_classes)
one_hot_test_labels = to_categorical(test_labels, num_classes)

model = models.Sequential()
model.add(Dropout(rate=0.1, input_shape=(10000,))) # add some randomness to prevent overfitting
model.add(layers.Dense(128, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

# using categorical_crossentropy as this is multi-class classification problem
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# training, 90% of training set used for training, 10% for validation
history = model.fit(train_data, one_hot_train_labels, batch_size=512, epochs=3, verbose=1, validation_split=0.1)

# evaluate model on the test data
score = model.evaluate(test_data, one_hot_test_labels, batch_size=512, verbose=1)

# this is how much our trained model was able tyo correctly predict topics of newswires from the content
print('Test accuracy', score[1])

# plot results
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(loss) + 1)

plt.figure(1)
plt.title('Validation accuracy')
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.figure(2)
plt.title('Accuracy of the train and validation sets')
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'bo', label='Training set')
plt.plot(epochs, val_acc, 'r', label='Validation')
plt.ylabel('Loss')
plt.legend()
plt.show()





