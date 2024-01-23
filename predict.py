import os
import torch
from FCN import fcn_resnet50
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


aux = False
classes = 20
weight_path = "./save_weights/model_29.pth"
img_path = "./image/96b0cb94gy1hknptckl9ej21hc0u0gqu.jpg"
assert os.path.exists(weight_path),f"weights{weight_path}not found"
assert os.path.exists(img_path),f"image{img_path}not found"
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

# 显示原始图像
plt.subplot(1, 2, 1)
plt.imshow(original)
plt.title('Original Image')
plt.axis('off')

# 显示分割掩码
plt.subplot(1, 2, 2)
plt.imshow(predicted_class, cmap='jet')
plt.title('Segmentation Mask')
plt.axis('off')

plt.show()