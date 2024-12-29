import os
import numpy as np
import time
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cv2
from sklearn.decomposition import PCA


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
            img = img.flatten()
            data.append(img)
            labels.append(label)
    return np.array(data), np.array(labels)


X, y = load_data(data_path, categories)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)


kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))
gpc = GaussianProcessClassifier(kernel=kernel)

start_time = time.time()
gpc.fit(X_train, y_train)
training_time = time.time() - start_time

start_time = time.time()
y_pred = gpc.predict(X_test)
test_time = time.time() - start_time

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
conf_matrix = confusion_matrix(y_test, y_pred)

print("Training Time: {:.2f} seconds".format(training_time))
print("Test Time: {:.2f} seconds".format(test_time))
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("F1 Score: {:.2f}".format(f1))
print("Confusion Matrix:\n", conf_matrix)
