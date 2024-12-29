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

# 设置路径
data_path = 'Train'  # 数据集路径
categories = ['albumin', 'collagen', 'pepsin', 'pancreatin']  # 4种蛋白质文件夹名称

# 准备数据
def load_data(data_path, categories):
    data = []
    labels = []
    for label, category in enumerate(categories):
        category_path = os.path.join(data_path, category)
        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            # 读取图片并调整为224x224
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))  # 调整大小
            img = img / 255.0  # 归一化图像像素值到[0,1]
            data.append(img)
            labels.append(label)
    return np.array(data), np.array(labels)

# 加载数据
X, y = load_data(data_path, categories)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 由于BiGRU要求输入3D数据，我们将数据调整为形状 (样本数, 时间步数, 特征数)
# 对于图像来说，可以将224x224的每个像素视为一个时间步，或者展平图像来作为每个时间步的输入
X_train = X_train.reshape((X_train.shape[0], 224, 224 * 3))  # 224时间步，每个时间步有224 * 3特征
X_test = X_test.reshape((X_test.shape[0], 224, 224 * 3))  # 同样的处理测试数据

# 定义BiGRU模型
model = Sequential()

# 添加双向GRU层
model.add(Bidirectional(GRU(32, return_sequences=False), input_shape=(224, 224 * 3)))

# 添加全连接层用于分类
model.add(Dense(12, activation='relu'))
model.add(Dense(4, activation='softmax'))  # 4个类别

# 编译模型
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练模型并计算训练时间
start_time = time.time()
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
training_time = time.time() - start_time

# 测试模型并计算测试时间
start_time = time.time()
y_pred = model.predict(X_test)
test_time = time.time() - start_time

# 获取预测标签
y_pred_classes = np.argmax(y_pred, axis=1)

# 计算各项指标
accuracy = accuracy_score(y_test, y_pred_classes)
recall = recall_score(y_test, y_pred_classes, average='macro')
precision = precision_score(y_test, y_pred_classes, average='macro')
f1 = f1_score(y_test, y_pred_classes, average='macro')
conf_matrix = confusion_matrix(y_test, y_pred_classes)

# 输出结果
print("Training Time: {:.2f} seconds".format(training_time))
print("Test Time: {:.2f} seconds".format(test_time))
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("F1 Score: {:.2f}".format(f1))
print("Confusion Matrix:\n", conf_matrix)
