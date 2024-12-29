import os
import numpy as np
import time
import tensorflow as tf
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Bidirectional, Flatten
from tensorflow.keras.optimizers import Adam

data_path = 'Train'
categories = ['albumin', 'collagen', 'pepsin', 'pancreatin']

def load_data(data_path, categories):
    data = []
    labels = []
    for label, category in enumerate(categories):
        category_path = os.path.join(data_path, category)
        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))
            img = img / 255.0
            data.append(img)
            labels.append(label)
    return np.array(data), np.array(labels)

X, y = load_data(data_path, categories)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train = X_train.reshape((X_train.shape[0], 224, 224 * 3))
X_test = X_test.reshape((X_test.shape[0], 224, 224 * 3))

model = Sequential()

model.add(Bidirectional(GRU(32, return_sequences=False), input_shape=(224, 224 * 3)))

model.add(Dense(12, activation='relu'))
model.add(Dense(4, activation='softmax'))  

model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

start_time = time.time()
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
training_time = time.time() - start_time

start_time = time.time()
y_pred = model.predict(X_test)
test_time = time.time() - start_time

y_pred_classes = np.argmax(y_pred, axis=1)

accuracy = accuracy_score(y_test, y_pred_classes)
recall = recall_score(y_test, y_pred_classes, average='macro')
precision = precision_score(y_test, y_pred_classes, average='macro')
f1 = f1_score(y_test, y_pred_classes, average='macro')
conf_matrix = confusion_matrix(y_test, y_pred_classes)

print("Training Time: {:.2f} seconds".format(training_time))
print("Test Time: {:.2f} seconds".format(test_time))
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("F1 Score: {:.2f}".format(f1))
print("Confusion Matrix:\n", conf_matrix)
