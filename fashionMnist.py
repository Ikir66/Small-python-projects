#fashion mnist dataset https://github.com/zalandoresearch/fashion-mnist 
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

from numpy.random import seed
seed(1)
tf.random.set_seed(1)

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(f'Number of training examples: {train_images.shape}')
print(f'Number of test examples: {test_images.shape}')

X_train=train_images[:-5000]/255.0
y_train=train_labels[:-5000]

X_valid=train_images[-5000:]/255.0
y_valid=train_labels[-5000:]

test_images=test_images/255.0

print(f'Number of training examples: {X_train.shape[0]}')
print(f'Number of validation examples: {X_valid.shape[0]}')

X_train=X_train.reshape(-1,28,28,1)
X_valid=X_valid.reshape(-1,28,28,1)

#model architecture
model = tf.keras.models.Sequential([
    #input block
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[28,28,1]),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2),
    #2nd block
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2),
    #flatten
    tf.keras.layers.Flatten(),
    #dense layer
    tf.keras.layers.Dense(128, activation="relu"),
    #output layer
    tf.keras.layers.Dense(10, activation="softmax")
])
model.summary()

model.compile(loss="sparse_categorical_crossentropy",
              optimizer='RMSprop',
              metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=20, batch_size=32,
validation_data=(X_valid, y_valid))

if int(tf.__version__.split('.')[0]) > 1:
    acc_key = 'accuracy'
else:
    acc_key = 'acc'

acc      = history.history[acc_key]
val_acc  = history.history['val_'+acc_key]
loss     = history.history['loss']
val_loss = history.history['val_loss']
epochs   = range(1,len(acc)+1)

plt.plot(epochs, acc,  label='Training accuracy')
plt.plot(epochs, val_acc,  label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xticks(epochs)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend();

test_loss, test_accuracy = model.evaluate(test_images.reshape(-1,28,28,1), test_labels, batch_size=128, verbose=2)
print('Accuracy on test dataset:', test_accuracy)
