import torch
from torchvision.utils import save_image
from Unet import UNet
from data import *
import matplotlib.pyplot as plt
from PIL import ImageOps

net = UNet().cuda()

weights = 'C:\\Users\\86188\\.conda\\yyy\\pythonProject1\\segment\\unet\\save_weights\\resnet50.pth'
if os.path.exists(weights):
    net.load_state_dict(torch.load(weights))
    print('successfully')
else:
    print('no loading')

# 在这里输入图片地址，例如：E:/Workspace/Unet/test_image/000033.jpg
_input = 'C:\\Users\\86188\\.conda\\yyy\\pythonProject1\\segment\\fcn\\image\\img.png'
img = keep_image_size_open(_input)
img_data = transform(img).cuda()
print(img_data.shape)
# 在保存图像之前，创建 'result/' 目录
os.makedirs('result/', exist_ok=True)
img_data = torch.unsqueeze(img_data, dim=0)  # 增加一个batch的维度
out = net(img_data)
# 保存图像为 PNG 格式
save_image(out, 'result/result.png', format='png')

# 加载原图和掩膜
original_img = Image.open(_input)
mask_img = Image.open('result/result.png')

# 裁剪掩膜到与原图相同的大小
mask_img = ImageOps.fit(mask_img, original_img.size, method=0, bleed=0.0, centering=(0.5, 0.5))

# 显示原图
plt.subplot(1, 2, 1)
plt.imshow(original_img)
plt.title('Original Image')
plt.gca().set_aspect('auto')

# 显示裁剪后的掩膜灰度图
plt.subplot(1, 2, 2)
plt.imshow(mask_img, cmap='gray')
plt.title('Mask (Gray)')
plt.gca().set_aspect('auto')

plt.show()


