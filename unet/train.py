from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from segment.fcn.train import get_transform
from segment.fcn.dataset import VOCSegmentation
from segment.fcn.eval import evaluate

from data import MyDataset
from Unet import UNet
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

weight_path = 'C:\\Users\\86188\\.conda\\yyy\\pythonProject1\\segment\\unet\\save_weights\\resnet50.pth'

data_path = 'D:\VOCtrainval_11-May-2012\VOCdevkit\VOC2012'
save_path = 'train_image'


def calculate_iou(output, target):
    # 转换为布尔掩码
    output = (output > 0.5).float()
    target = (target > 0.5).float()

    # 逻辑 AND 和 OR 操作替代位 AND 和 OR 操作
    intersection = (output.logical_and(target)).float().sum()
    union = (output.logical_or(target)).float().sum()

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()  # 假设你想返回一个 Python 浮点数


if __name__ == '__main__':
    batch_size = 4
    data_loader = DataLoader(MyDataset(data_path), batch_size=batch_size, shuffle=True)
    model = UNet()
    net = model.to(device)
    pretrained_dict = torch.load(weight_path)
    model_dict = net.state_dict()

    val_dataset = VOCSegmentation("D:\VOCtrainval_11-May-2012\VOCdevkit",
                                  year="2012",
                                  transforms=get_transform(train=False),
                                  txt="val.txt")
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            num_workers=num_workers,
                            pin_memory=True,
                            collate_fn=val_dataset.collate_fn)
    # Filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # Update model_dict with pretrained_dict
    model_dict.update(pretrained_dict)

    # Load the updated model_dict into the model
    net.load_state_dict(model_dict, strict=False)

    opt = optim.Adam(net.parameters())
    loss_fun = nn.BCELoss()

    n_epoch = 30
    total_iou = 0.0
    for epoch in range(0, n_epoch):
        ss_time = time.time()
        start_time = time.time()  # 记录训练开始时间
        total_loss = 0.0

        for i, (image, segment_image) in enumerate(data_loader):
            image, segment_image = image.to(device), segment_image.to(device)

            out_image = net(image)
            train_loss = loss_fun(out_image, segment_image)
            total_loss += train_loss.item()
            opt.zero_grad()
            train_loss.backward()
            opt.step()
            iou = calculate_iou((out_image > 0.5).float(), segment_image)
            total_iou += iou

            if i % 100 == 0:
                end_time = time.time()  # 记录每100个批次结束时间
                elapsed_time = end_time - start_time  # 计算每100个批次的训练时间差
                print(f'{epoch}-{i}-train_loss ===>> {train_loss.item()}')
                print(len(data_loader))

        torch.save(net.state_dict(), weight_path)
        elapsed_time = time.time()-ss_time
        print(f'Epoch {epoch} - Mean Loss: {total_loss / len(data_loader)}, Mean IoU: {total_iou / len(data_loader)}')
