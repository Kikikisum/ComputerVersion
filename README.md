## **1.背景**

### 1.1 图像分割

图像分割是图像处理和计算机视觉领域中的关键任务，其应用涵盖场景理解、医学图像分割、机器人感知、视频监控、增强现实和图像压缩等多个领域。本文着重回顾了基于深度学习模型的图像分割方法。

大多数图像分割算法都基于图像中灰度值的不连续性和相似性质。在前者中，算法以灰度突变为基础，实现对图像的分割，如图像边缘分割。这类算法假设图像不同区域的边界彼此之间存在显著的差异，并与背景有着明显区分，因此可以利用基于灰度的局部不连续性来进行边界检测。后者则是根据一组预定义的准则将图像分割为相似区域的方法，其中包括阈值处理、区域生长、区域分裂和区域聚合等。以下将对每一类算法进行详细说明。

图像分割主要分为以下三类：

- **语义分割**：对图像的像素点进行分类，就是把图像中每个像素赋予一个类别标签，预测结果为掩膜。
- **实例分割**：只需要找到图中物体的边缘轮廓。
- **全景分割**：语义分割和实例分割的结合。归于同一类别的像素如果存在多个实例，则会使用不同颜色进行区**别**。

### 1.2 PASCAL VOC数据集

> 本模型的数据集为VOC 2012，下载自http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html

#### 1.2.1 简介

PASCAL VOC挑战赛(The PASCAL Visual Object Classes ）是一个世界级的计算机视觉挑战赛，PASCAL全称:Pattern Analysis, Statical Modeling and Computational Learning，是一个由欧盟资助的网络组织。PASCAL VOC挑战赛主要包括以下几类:图像分类(Object Classification),目标检测(Object Detection)，目标分割(Object Segmentation)，动作识别(Action Classification)等。

PASCAL VOC数据集共有4大类：Vehicles ,Household, Animals, person

该四类又分成了20小类：

```
├──	Vehicles			├── Household				├──Animals				Person
	├──4-wheeled			├──Furniture				├──Domestic
	│	├──Car				│	├──Seating				│	├──Cat
	│	├──Bus				│		├──Chair			│	├──Dog
	├──2-wheeled			│		├──Sofa				├──Farmyard
	│	├──Bicycle			│	├──Dining table			│	├──Cow
	│	├──Motorbike		├──TV/monitor				│	├──Horse
	├──Aeroplane			├──Bottle					│	├──Sheep
	├──Boat					├──Potted plant				├──Bird
	├──Train
```

#### 1.2.2 目录结构

以VOC 2012为例：

```
VOCdevkit
    └── VOC2012
         ├── Annotations               所有的图像标注信息(XML文件)
         ├── ImageSets    
         │   ├── Action                人的行为动作图像信息
         │   ├── Layout                人的各个部位图像信息
         │   │
         │   ├── Main                  目标检测分类图像信息
         │   │     ├── train.txt       训练集 (5717)
         │   │     ├── val.txt         验证集 (5823)
         │   │     ├── trainval.txt    训练集+验证集 (11540)
         │   │
         │   └── Segmentation          目标分割图像信息(子目录结构同上)
         │ 
         ├── JPEGImages                所有图像文件
         ├── SegmentationClass         语义分割png图（基于类别）
         └── SegmentationObject        实例分割png图（基于目标）
```

### 1.3 评估指标

> 对代码中使用的评估指标进行介绍

#### 1.3.1 混淆矩阵

在机器学习领域，特别是统计分类问题中，**混淆矩阵（confusion matrix）**是一种特定的表格布局，用于可视化算法的性能，矩阵的每一行代表实际的类别，而每一列代表预测的类别。

- IOU：前景目标交并比
  $$
  I_{ou}=\frac{A∩B}{A∪B}
  $$

- mIOU：每个类的IOU平均值

- acc：准确率

#### 1.3.2 优化目标

**交叉熵损失**

zk表示网络输出，f(zk)表示概率
$$
f(z_{k})=\frac{e^{z_k}}{\sum\limits{e^{z_j}}},l(y,z)=-\sum\limits_{k=0}^{C}y_clog(f(z_k))
$$
分割损失即所有像素分类损失的累加

可用于多类别，正负样本梯度稳定

**dice损失**

P,G分别表示预测和真值，定义Dice相似度:
$$
S=2\frac{|P∩G|}{|P|+|G|}
$$

$$
S1=\frac{\sum_Np_ig_i}{\sum_{Np_i^2}+\sum_{Ng_i^2}}
$$

$$
S2=\frac{2\sum_{N}p_ig_i}{\sum_{N}p+\sum_Ng_i}
$$

dice损失D(p,g)=1−S

**缺点**

- 损失可能不稳定
- 专注正样本，适合小目标

## 2.**方法发展**

### 2.1 FCN

#### 2.1.1**特点**

- 在CNN的基础上，将最后的全连接层替换为卷积层，输出值从一维值替换为二维图像。

- 可接受任意尺寸的输入图像
- 利用反卷积层对最后一个卷积层的特征图进行上采样，使其恢复到输入图像相同的尺寸
- 跳跃连接

#### 2.1.2 上采样的实现

- 双线性插值：只是根据已知的像素点对未知的点进行预测估计，从而可以扩大图像的尺寸，达到上采样的效果
- 转置卷积：feature maps补0，然后做卷积操作
- 反池化：在空隙中填充 0

#### 2.1.3 跳跃连接

> 比较FCN-32s、FCN-16s以及FCN-8s，效果：FCN-32s < FCN-16s < FCN-8s，**使用多层feature融合有利于提高分割准确性**。

- 思路就是将**不同池化层**的结果进行**上采样**，然后结合这些结果来优化输出

- 大多数的上采样是**采用双线性插值**来快速实现上采样。

- 不采取反卷积的方法，**反卷积的效率不高且效果不好，计算量大**

### 2.2 U-Net

#### 2.2.1 特点

UNet网络由一个收缩路径和一个扩展路径组成，收缩路径与扩展路径相对应的网络层之间具有跳跃连接结构。

U-Net算法架构为**两次3x3的卷积**之后进行**一次池化**，总共**4次下采样**，因此，相对应的进行了**4次上采样**，因此形成了一个形似**U型**的网络架构,将每一次对应的上采样与下采样的特征图对应**拼接**，以确保输出的特征图**误差最小**

使用**镜面翻转**的方式对数据进行增强，如果需要预测边界上附近的图像时，该网络架构会将边界的一部分进行翻转（人眼的镜像效果）

#### 2.2.2 不足

- 必须训练每个patch，patch之间重叠存在很多**冗余**，**增加训练时间，降低效率**，用一张图片训练多次，可能会导致**过度拟合**
- **定位准确性和获取上下文信息不可兼得**，越大的patch需要更多的pooling层来达到这个大小，pooling层越多，丢失的信息越多，小的的patch只能看到局部信息

#### 2.2.3 使用场景

U-Net是针对生物医学图像进行设计的，U型结构更有助于捕获图像中的细节信息，同时保留全局上下文信息。

### 2.3 DeepLabV3

#### 2.3.1 特点

- 空间金字塔池化：为了多尺度物体的语义分割，使用**不同采样率**和**多种视野的卷积核**，以捕捉多尺度对象
- 空洞卷积：在卷积核中加入一些**零元素**，感受野中有数据的像素被零元素区隔开，**扩大了感受野**，并且零元素**不参与**参数计算

- **通过组合DCNN和概率图模型（CRF），改进分割边界结果**

#### 2.3.2 ASPP结构

该模型结构采用了四个具有不同rate进行空洞卷积，对不同尺度的特征图进行重采样对于任意尺度的区域进行分类是非常有效的（该结构包含了Batch Normalization）。在image pooling部分，主要由1x1和3x3的卷积组成，并且3x3卷积采用不同的rate从而得到是三个不同的输出特征图。

**随着atrous rate的变化，在一定程度上将会导致卷积核退化为为1x1卷积。**

DeepLabV3的ASPP结构相比于DeepLabV2的ASPP结构要更加的“丰富”，其中主要添加的有：激活函数（ReLU）和批量归一化，以及多出来的一个分支。该分支的结构细节是，首先是进行全局的平均池化，之后经过一个1x1且步长为1的标准卷积，之后就是激活函数（ReLU）和批量归一化，最后采用线性插值。

## 3.方法介绍

> 主要学习自论文：https://arxiv.org/abs/1411.4038
>
> 代码存放在：https://github.com/Kikikisum/ComputerVersion/tree/latest_branch

### 3.1 问题陈述

> 权重文件路径：https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth

利用PASCAL VOC2012训练集，采用fcn_resnet50_coco-1167a1af.pth进行载入resnet50 backbone预训练权重，来实现语义分割

### 3.2 模型架构

#### 3.2.1 Resnet50

> 该部分内容位于model.py文件内

采用Resnet50作为Backbone部分，即为下采样部分。

**残差结构**

在正常的神经网络，增加一条short cut支路，称为高速公路。这条支路使得神经网络不是简单的输出卷积后的值，而是**卷积后和卷积前的值的叠加值**。

当卷积层数过多时，很容易出现梯度消失现象，使得反向传播无法进行。

高速公里使得输入数据无损地通过。如果左侧卷积层学习到的数据不够好，那么叠加上无损通过的原始数据，依然保留了原始数据，不至于丢掉原始数据。**避免了梯度消失。**

输入层->最大池化层->Layer1->Layer2->Layer3->Layer4->全局平均池化层->全连接层

![image-20240123234818427](C:\Users\86188\AppData\Roaming\Typora\typora-user-images\image-20240123234818427.png)

**输入层**

- 一个7*7的卷积层，步幅为2，进行初始特征提取
- 接着使用bn和ReLU激活函数

**池化层**

- 使用最大池化层，kernel大小为3*3，步幅为2，进行下采样

**Bottleneck**

- Bottleneck则是一个1x1，3x3，1x1共三个卷积核组成

- 四个**Bottleneck**组成一个Resnet模块(layer1、layer2、layer3、layer4)

- ResNet-50 具有不同深度的四个模块，每个模块包含的残差块数量不同

**全局平均池化层**

- 使用自适应全局平均池化层，将特征图大小降至 1x1

**全连接层**

使用一个全连接层，将输出的特征图压缩为具体类别的预测结果

#### 3.2.2 FCNHead

> 该部分内容位于FCN.py文件内

采用FCNHead类将Backbone的输出转化为最终的语义分割图像，即为下采样部分。

1. 一个3*3的卷积层，使用bn和ReLU激活函数
2. Dropout随机丢弃一定比例的神经元，防止过拟合
3. 一个1*1的卷积层，把特征映射到输出通道

![image-20240123234858508](C:\Users\86188\AppData\Roaming\Typora\typora-user-images\image-20240123234858508.png)

#### 3.2.3 aux_classifier

aux是辅助分类器，采取和FCNHead相同的架构(不重复赘述)，aux接收的是Resnet50的Layer3的特征，通过中间层的监督，提高对局部结构的识别功能。

## 4.实验结果

### 4.1 模型运行的指标

> 因为进行了30个epoch，仅展示epoch 0和epoch 29，其他数据在results20240122-203324.txt查看

```
[epoch: 0]
train_loss: 1.0333
lr: 0.000100
global correct: 93.0
average row correct: ['96.5', '90.0', '76.7', '81.9', '82.7', '59.4', '94.6', '81.9', '92.8', '56.6', '78.3', '63.4', '83.0', '75.6', '88.4', '94.9', '63.0', '85.5', '68.0', '86.1', '84.3']
IoU: ['93.0', '84.2', '38.3', '77.3', '66.8', '55.3', '86.8', '72.0', '79.5', '40.6', '69.1', '52.2', '67.7', '70.8', '77.3', '86.4', '53.1', '72.0', '50.7', '81.8', '71.0']
mean IoU: 68.9
```

------

```
[epoch: 29]
train_loss: 0.5226
lr: 0.000000
global correct: 90.8
average row correct: ['97.3', '75.0', '66.2', '54.1', '60.0', '80.1', '43.4', '86.7', '82.7', '58.1', '17.4', '73.1', '72.2', '76.8', '76.0', '93.8', '65.1', '77.1', '70.4', '63.9', '88.9']
IoU: ['92.8', '73.1', '50.6', '52.5', '50.8', '69.5', '42.2', '66.1', '66.2', '40.2', '17.0', '58.3', '48.6', '50.6', '70.5', '87.3', '57.3', '54.1', '50.8', '54.3', '70.0']
mean IoU: 58.2
```

- 在 Epoch 29，训练损失相对较低，为 0.5226，表明模型在训练数据上的拟合效果较好。
- 全局准确率为 90.8%，表示模型在整个数据集上的分类准确率较高。
- 平均交并比（mean IoU）为 58.2%，表明模型在图像分割任务中的性能较为中等。

### 4.2 指标图像

#### 4.2.1 损失曲线

![image-20240124002242689](C:\Users\86188\AppData\Roaming\Typora\typora-user-images\image-20240124002242689.png)

#### 4.2.2 mean IoU曲线

![image-20240124003110973](C:\Users\86188\AppData\Roaming\Typora\typora-user-images\image-20240124003110973.png)

### 4.3 预测

利用训练Epoch29保存的权重文件进行预测，从VOC中随机挑选一张照片进行预测，同时展示原图和掩码

![image-20240124010816026](C:\Users\86188\AppData\Roaming\Typora\typora-user-images\image-20240124010816026.png)

同时挑选了一张动漫图片

![image-20240124011148584](C:\Users\86188\AppData\Roaming\Typora\typora-user-images\image-20240124011148584.png)

## 5.方法修改与对比

### 5.1 方法修改

将FCN的backbone分别用三个模型来实现，分别为resnet50、resnet101、resnet152三个模型，除开FCN的backbone部分，其他部分不进行修改。

### 5.2 方法对比

#### 5.2.1 Loss曲线

![image-20240201000225605](C:\Users\86188\AppData\Roaming\Typora\typora-user-images\image-20240201000225605.png)

- Resnet101的Loss曲线位于三个曲线的最低部分，在Loss最低位置来看，Resnet101的效果最好
- 但Resnet152的Loss曲线的前期的斜率最大，在降低Loss来看，Resnet152的效果最好
- Resnet152和Resnet50后期的斜率接近0，曲线平缓，可能存在过拟合的情况

> 在调试了好几次Resnet152模型的学习率和优化器后发现，Loss一直为较高的值，这个问题还在想解决方案

#### 5.2.2 mIoU曲线

> 三个模型都使用了pytorch官方的权重文件进行预训练
>
> 预训练后，Resnet50和Resnet101初期都展示了较高的mIoU值
>
> 但Resnet152出现权重文件缺失模型的部分层的权重，这个不清楚原因

![image-20240201000942923](C:\Users\86188\AppData\Roaming\Typora\typora-user-images\image-20240201000942923.png)

- Resnet50的mIoU随着训练次数增加，逐渐下降
- Resnet101的mIoU相对平稳
- Resnet152的mIoU增加速度较快

#### 5.2.3 同一图片的分割效果

原图为：

![img](C:\Users\86188\.conda\yyy\pythonProject1\segment\fcn\image\img.png)

Resnet50为：

![image-20240201011136675](C:\Users\86188\AppData\Roaming\Typora\typora-user-images\image-20240201011136675.png)

Resnet101为：

![101plane](C:\Users\86188\.conda\yyy\pythonProject1\segment\fcn\image\101plane.png)

Resnet152为：

![image-20240201011218797](C:\Users\86188\AppData\Roaming\Typora\typora-user-images\image-20240201011218797.png)

Resnet101最接近原图的掩膜

### 5.3 不同模型对比基于同一数据集

#### 5.3.1 Loss曲线

![image-20240201013628379](C:\Users\86188\AppData\Roaming\Typora\typora-user-images\image-20240201013628379.png)

- **DeepLabV3**从初始时期开始下降，逐渐趋于稳定。在中后期，损失下降速度减缓，但整体上呈现出较为平滑的趋势。
- **FCN**训练损失一开始迅速下降，然后逐渐趋于平稳。相对于DeepLabV3，FCN的损失值整体较低，说明模型在训练数据上取得了较好的拟合。
- **Unet**变化平稳。

#### 5.3.1 mIoU曲线

![](C:\Users\86188\AppData\Roaming\Typora\typora-user-images\image-20240201013551056.png)

- **DeepLabV3**在训练初期迅速上升，之后呈现出相对平稳的趋势。

- **FCN **mIoU 的上升趋势比 DeepLabV3 缓慢，但随着训练的进行，取得了更高的 mIoU 值。

- **mIoU** 曲线的波动相对较大，尤其是在训练后期。
