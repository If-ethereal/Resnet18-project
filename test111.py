from fastai.vision.all import *

# 下载并解压CIFAR-10数据集
path = untar_data(URLs.CIFAR)

# 创建数据加载器
dls = ImageDataLoaders.from_folder(path, valid_pct=0.2, seed=42)

# 创建学习器并使用 resnet18
learner = cnn_learner(dls, models.resnet18, metrics=accuracy)
learner.fine_tune(1)
