import os
import torch
from FCN import fcn_resnet50
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


aux = False
classes = 20
weight_path = "D:\model_29.pth"
img_path = "image/96b0cb94gy1hknptckl9ej21hc0u0gqu.jpg"
assert os.path.exists(weight_path) ,f"weights{weight_path}not found"
assert os.path.exists(img_path), f"image{img_path}not found"
device = torch.device("cpu")
model = fcn_resnet50(aux=aux, num_classes=classes+1)
model.to('cuda')  # 将模型移动到 GPU，如果可用
weight_dict = torch.load(weight_path, map_location='cpu')['model']
for k in  list(weight_dict.keys()):
    if "aux" in k:
        del weight_dict[k]

model.load_state_dict(weight_dict)
original = Image.open(img_path)

data_transform = transforms.Compose([transforms.Resize(520),
                                     transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225))
                                    ])
img = data_transform(original)
image = img.unsqueeze(0)  # 添加批次维度
# 将输入移动到 GPU
image = image.to('cuda')

with torch.no_grad():
    image = image.to('cuda')  # 将输入移动到 GPU，如果可用
    output = model(image)

# 提取类别概率张量
predicted_class = torch.argmax(output['out'], dim=1).cpu().numpy()[0]

# 只保存掩膜
plt.imshow(predicted_class)  # 使用 'gray' cmap
plt.title('Segmentation Mask')
plt.axis('off')

# 保存图像
plt.savefig('50m.png')
plt.close()