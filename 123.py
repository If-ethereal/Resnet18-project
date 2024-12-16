from fastai import *
import multiprocessing

from fastai.torch_core import TensorImage
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
import fastai
import torch
print(fastai.__version__)
print(torch.__version__)

import matplotlib
from fastai.callback.tensorboard import TensorBoardCallback
from fastai.vision import models
from fastai.vision.learner import cnn_learner, vision_learner
from torchvision.models import resnet34

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from fastai.data.block import DataBlock, CategoryBlock
from fastai.data.transforms import get_image_files, RegexLabeller, RandomSubsetSplitter
from fastai.vision import *
from fastai.metrics import error_rate, accuracy
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from pathlib import Path
from functools import partial
from fastai.vision.augment import Resize
from fastai.vision.data import ImageBlock
from fastcore.basics import using_attr
y  = 'seg_train'
path = Path(y)
np.random.seed(40)
torch.manual_seed(42) #想解决学习率图像显示不一样的尝试1
data_block = DataBlock(
    blocks=[ImageBlock, CategoryBlock],
    get_items=get_image_files,
    item_tfms=Resize(150),
    splitter=RandomSubsetSplitter(train_sz=0.8,valid_sz=0.2, seed=42),  # 可以替换成其它分割方法
    get_y=lambda x: x.parent.name
    #et_y=using_attr(lambda x: Path(x).parent, 'name')

)
dls=data_block.dataloaders('seg_train',num_workers=0)
print(len(dls.train_ds), len(dls.valid_ds))
items = get_image_files('seg_train')
#for item in items[:5]:
#    print(item, Path(item).parent.name)
#items = get_image_files(path)
#print(f"Total images found: {len(items)}")
#for item in items[:5]:
#    print(item)
# 打印前一个训练数据的图像及其标签
for img, label in dls.train_ds:
    print(img, dls.vocab[label])  # 使用 vocab 获取类别名称
    break  # 只打印第一对，以避免输出过多
print(dls.vocab)  # 查看类别名称
#all_labels = [label for _, label in dls.train_ds]
#print(set(all_labels))  # 查看所有独特的标签

#all_labels = [item[1] for item in dls.train_ds]
#print(set(all_labels))  # 使用set()来查看所有独特的标签


dls.show_batch(max_n=9, figsize=(8,6))
plt.show()
learner = vision_learner(dls, models.resnet18, metrics=accuracy)
learner.load('my_model')

#learner = vision_learner(dls, models.efficientnet_b0, metrics=accuracy)


print("juanji:",learner.model)

# if __name__ == '__main__':
#learner.fine_tune(1)
#learner.save('my_model2')   # 模型训练部分
#print("模型已保存")


# learner.freeze() #想解决学习率图像显示不一样的尝试1
# learner.lr_find()
plt.show()
# if __name__ == '__main__':
#     learn = vision_learner(dls, models.resnet18, metrics=accuracy, model_dir=Path('./model'))
#     print(type(learn))
#     learn.unfreeze()
#     learn.fit_one_cycle(3,cbs=TensorBoardCallback(Path('modelll/suntext'),  trace_model=True))
#learn = vision_learner(dls, models.resnet18, metrics=error_rate)
# learn = cnn_learner(dls, models.resnet18, metrics=error_rate)
# learn.fine_tune(1)  # 训练5个周期
#
# 定义一个新的 DataBlock，用于加载新的测试集
# 定义一个新的 DataBlock，用于加载新的测试集
# 定义测试集路径
# 加载新验证集的图像
test_path = Path('seg_cartoontest')

# 获取图像文件列表
test_images = get_image_files(test_path)
print(f"Found {len(test_images)} test images.")

# 预处理这些图像，与训练集的预处理相同
test_dl = dls.test_dl(test_images, with_labels=True)

# 使用已训练的模型进行预测
preds, targs = learner.get_preds(dl=test_dl)

# 获取每个预测的标签和实际标签
predicted_labels = preds.argmax(dim=1)
actual_labels = targs

# 输出预测的标签和实际标签
for i, img_path in enumerate(test_images):
    predicted_label = dls.vocab[predicted_labels[i]]
    actual_label = img_path.parent.name  # 从文件路径中提取实际标签
    print(f"Image: {img_path.name}, Predicted: {predicted_label}, Actual: {actual_label}")
learner.show_results(dl=test_dl, max_n=len(test_images))

# 使用plt.show()来显示图像
plt.show()




# # 第四步：评估模型
learner.show_results()
plt.show()
# #混淆矩阵！！！
# preds, targs = learner.get_preds()

#
# # 获取验证集的预测和实际标签
# preds, targs = learner.get_preds()
#
# # 获取预测的类别（索引）
# pred_classes = preds.argmax(dim=1)
#
# # 获取混淆矩阵
# cm = confusion_matrix(targs, pred_classes)
#
# # 可视化混淆矩阵
# def plot_confusion_matrix(cm, labels, cmap=plt.cm.Blues):
#     plt.figure(figsize=(10,7))
#     sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=labels, yticklabels=labels, cbar=False)
#     plt.xlabel('Predicted Label')
#     plt.ylabel('True Label')
#     plt.title('Confusion Matrix')
#     plt.show()
#
# # 使用错误矩阵中的标签名
# labels = dls.vocab
# plot_confusion_matrix(cm, labels)
#
# # 错误分析：找到误分类的样本
# misclassified = []
# for i in range(len(pred_classes)):
#     if pred_classes[i] != targs[i]:  # 如果预测标签与实际标签不匹配
#         misclassified.append((dls.valid_ds.items[i], preds[i], targs[i]))
#
# # 打印一些误分类的样本
# for img, pred, target in misclassified[:5]:
#     print(f"Image: {img}, Predicted: {dls.vocab[pred.argmax()]}, Actual: {dls.vocab[target]}")
#
# # 可视化一些误分类的图像
# def show_misclassified_images(misclassified, n=9):
#     fig, axes = plt.subplots(3, 3, figsize=(10, 10))
#     for i, (img_path, pred, actual) in enumerate(misclassified[:n]):
#         img = cv2.imread(str(img_path))
#         axes[i//3, i%3].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#         axes[i//3, i%3].set_title(f"Pred: {dls.vocab[pred.argmax()]}\nActual: {dls.vocab[actual]}")
#         axes[i//3, i%3].axis('off')
#     plt.tight_layout()
#     plt.show()
#
# # 显示一些误分类的图像
# show_misclassified_images(misclassified)

##分类报告
from sklearn.metrics import classification_report

# 获取预测和实际标签
preds, targs = learner.get_preds()

# 生成分类报告
report = classification_report(targs, preds.argmax(dim=1), target_names=dls.vocab)
print(report)









#
# # 1. 提取卷积层的输出
# def register_hooks(learn):
#     """注册 hooks 用于提取特征图和梯度"""
#     # 获取卷积神经网络的最后一个卷积层
#     last_conv_layer = learn.model[0][6]
#
#     # 定义 hook 函数
#     def hook_fn(module, input, output):
#         # 保存输出特征图
#         learn.activations = output.detach()
#
#     # 注册 hook
#     hook = last_conv_layer.register_forward_hook(hook_fn)
#     return hook
#
# # 注册 hook
# import torch.nn.functional as F
#
# hook = register_hooks(learner)
#
# def generate_grad_cam(learn, img):
#     """生成 Grad-CAM"""
#     # 传递图片进入模型计算预测结果
#     img = img.unsqueeze(0).to(learn.dls.device)  # 增加batch维度
#     learn.model.eval()
#
#     # 计算模型的输出
#     output = learn.model(img)
#     pred_class = output.argmax(dim=1).item()  # 预测类别
#     pred_prob = output[0, pred_class].item()  # 预测类别的概率
#
#     # 计算预测类别的梯度
#     learn.model.zero_grad()
#     output[0, pred_class].backward()
#
#     # 获取最后卷积层的梯度
#     grads = learn.model[0][6].weight.grad.detach()
#
#     # 获取最后卷积层的输出特征图
#     activations = learn.activations
#
#     # 计算权重（将梯度加权并平均）
#     weights = grads.mean(dim=(2, 3), keepdim=True)
#
#     # 计算加权特征图（Grad-CAM）
#     grad_cam = torch.sum(weights * activations, dim=1, keepdim=True)
#
#     # 使用 ReLU 激活函数
#     grad_cam = F.relu(grad_cam)
#
#     # 归一化
#     grad_cam = grad_cam.squeeze().cpu().numpy()
#     grad_cam = np.maximum(grad_cam, 0)
#     grad_cam = cv2.resize(grad_cam, (img.shape[2], img.shape[3]))
#     grad_cam -= grad_cam.min()
#     grad_cam /= grad_cam.max()
#
#     return grad_cam, pred_class, pred_prob
#
#
#
#
#
# def plot_grad_cam(learn, img, ax=None):
#     """显示 Grad-CAM 可视化结果"""
#     grad_cam, pred_class, pred_prob = generate_grad_cam(learn, img)
#
#     # 获取输入图像
#     img = img.permute(1, 2, 0).cpu().numpy()
#
#     # 显示输入图像和 Grad-CAM
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(10, 10))
#
#     ax.imshow(img)
#     ax.imshow(grad_cam, cmap='jet', alpha=0.5)  # 使用热力图覆盖原图
#     ax.set_title(f'Pred: {learn.dls.vocab[pred_class]} | Prob: {pred_prob:.2f}')
#     ax.axis('off')
#
#     plt.show()
#
#
# # 选择一个样本图像并显示 Grad-CAM
# img, label = dls.train_ds[0]  # 选择训练集的第一个样本
# plot_grad_cam(learner, img)

# ##Class Activation Maps (CAM)
# # 假设 'learner' 是你已经训练好的模型
# hook = learner.model[0][6].register_hooks(lambda m, i, o: o)
#
# # 随机选择一张图片
# img, label = dls.valid_ds[0]
#
# # 将图像输入模型
# pred = learner.model(img.unsqueeze(0))  # 添加批次维度
#
# # 获取卷积层输出的特征图（feature map）
# feature_map = hook.stored[0]
#
# # 计算每个特征图的重要性（通过加权平均池化）
# weights = learner.model[1].weight  # 获取模型最后一层的权重
# weight_for_class = weights[label]  # 获取对应标签的权重
#
# # 计算 class activation map (CAM)
# cam = torch.matmul(feature_map.squeeze(), weight_for_class)  # 用权重和特征图计算
#
# # 对CAM进行归一化
# cam = cam - cam.min()
# cam = cam / cam.max()
#
# # 显示 CAM
# plt.imshow(cam.detach().cpu().numpy(), cmap='jet')
# plt.colorbar()
# plt.show()


