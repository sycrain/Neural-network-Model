import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from net import vgg16
from torch.utils.data import DataLoader
from data import *
import h5py
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 数据集
annotation_path = 'cls_Train.txt'
with open(annotation_path, 'r') as f:
    lines = f.readlines()
np.random.seed(10101)
np.random.shuffle(lines)
np.random.seed(None)
num_val = int(len(lines) * 0.2)
num_train = len(lines) - num_val

input_shape = [224, 224]
train_data = DataGenerator(lines[:num_train], input_shape, True)
val_data = DataGenerator(lines[num_train:], input_shape, False)
val_len = len(val_data)
print("数据集总数:%d" % (num_train + val_len))
print("训练集数:%d" % (num_train))
print("测试集数：%d" % (val_len))

# 加载数据
gen_train = DataLoader(train_data, batch_size=64)
gen_test = DataLoader(val_data, batch_size=64)

# 构建网络
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
net = vgg16(True, progress=True, num_classes=4)
net.to(device)

# 选择优化器和学习率的调整方法
lr = 0.00015
optim = torch.optim.Adam(net.parameters(), lr=lr)
sculer = torch.optim.lr_scheduler.StepLR(optim, step_size=3)

# 训练
epochs = 10
x = []
y1 = []
y2 = []
y3 = []
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(epochs):
    total_train = 0
    correct_train = 0
    net.train()
    for data in gen_train:
        img, label = data
        img = img.to(device)
        label = label.to(device)
        optim.zero_grad()
        output = net(img)
        train_loss = nn.CrossEntropyLoss()(output, label).to(device)
        train_loss.backward()
        optim.step()
        total_train += train_loss.item()
        correct_train += (output.argmax(1) == label).sum().item()
    sculer.step()
    train_losses.append(total_train)
    train_accuracies.append(correct_train / num_train)

    total_test = 0
    correct_test = 0
    all_preds = []
    all_labels = []
    net.eval()
    for data in gen_test:
        img, label = data
        img = img.to(device)
        label = label.to(device)
        with torch.no_grad():
            out = net(img)
            test_loss = nn.CrossEntropyLoss()(out, label).to(device)
            total_test += test_loss.item()
            correct_test += (out.argmax(1) == label).sum().item()
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    test_losses.append(total_test)
    test_accuracies.append(correct_test / val_len)

    print("训练集上的损失：{}".format(total_train))
    print("测试集上的损失：{}".format(total_test))
    print("训练集上的精度：{:.1%}".format(correct_train / num_train))
    print("测试集上的精度：{:.1%}".format(correct_test / val_len))

    x.append(epoch)
    y1.append(total_train)
    y2.append(total_test)
    y3.append(correct_test / val_len)
    torch.save(net.state_dict(), "../PTH/Dbz{}.pth".format(epoch + 1))
    print("模型已保存")

# 生成并保存最后一轮的混淆矩阵
class_names = ['Albumin', 'Collagen', 'Pepsin', 'Pancreatin']

cm = confusion_matrix(all_labels, all_preds)
print(f"混淆矩阵形状: {cm.shape}")
print(f"类别数量: {len(class_names)}")
if cm.shape[0] == len(class_names):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation='vertical', values_format='.0f')
    plt.tight_layout()
    for text in ax.texts:
        text.set_fontsize(16)
    plt.xticks(rotation=45)
    #plt.savefig("混淆矩阵_最终轮次_1.png", dpi=600)
    plt.show()
else:
    print("混淆矩阵的形状和类别数量不匹配，请检查类别标签和预测结果。")

x = np.array(x)
y1 = np.array(y1)
y2 = np.array(y2)
y3 = np.array(y3)
plt.plot(x, y1, color='green', linewidth=1, linestyle='-', marker='o', markersize=2)
plt.xlabel('epochs')
plt.ylabel('train_loss')
plt.show()
plt.plot(x, y2, color='green', linewidth=1, linestyle='-', marker='o', markersize=2)
plt.xlabel('epochs')
plt.ylabel('test_loss')
plt.show()
plt.plot(x, y3, color='green', linewidth=1, linestyle='-', marker='o', markersize=2)
plt.xlabel('epochs')
plt.ylabel('test_accuracy')
plt.show()

# 保存损失和精度数据
np.savetxt("..\Results\loss_acc/train_losses_1.txt", train_losses, fmt='%.3f')
np.savetxt("..\Results\loss_acc/test_losses_1.txt", test_losses, fmt='%.3f')
np.savetxt("..\Results\loss_acc/train_accuracies_1.txt", train_accuracies, fmt='%.3f')
np.savetxt("..\Results\loss_acc/test_accuracies_1.txt", test_accuracies, fmt='%.3f')
