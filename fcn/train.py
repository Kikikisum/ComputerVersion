import os
import time
import datetime
from torch.utils.data import DataLoader
import torch
import argparse
from segment.fcn.utils import RandomResize, RandomHorizontalFlip, RandomCrop, Compose, ToTensor, Normalize
from segment.fcn.FCN import fcn_resnet50
from segment.fcn.dataset import VOCSegmentation
from segment.fcn.eval import create_lr_scheduler, train_one_epoch, evaluate
from torch.multiprocessing import freeze_support


class SegmentationPresetTrain:
    """
    #训练过程中图像预处理方法
    """

    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)  # 520*0.5=260
        max_size = int(2.0 * base_size)  # 520*2=1024

        trans = [RandomResize(min_size, max_size)]  # 随机选取数值
        if hflip_prob > 0:
            # 随机水平翻转大于0
            trans.append(RandomHorizontalFlip(hflip_prob))  # 水平翻转
        trans.extend([
            RandomCrop(crop_size),  # 随机裁剪
            ToTensor(),
            Normalize(mean=mean, std=std),  # 标准化
        ])

        self.transforms = Compose(trans)  # 将预处理方法进行打包

    def __call__(self, img, target):
        return self.transforms(img, target)  # transforms方法是在call中被调用的


class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = Compose([
            RandomResize(base_size, base_size),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train):
    base_size = 520
    crop_size = 480
    # 后者是验证集的图片大小
    return SegmentationPresetTrain(base_size, crop_size) if train else SegmentationPresetEval(base_size)


def create_model(aux, num_classes, pretrain=True):
    model = fcn_resnet50(aux=aux, num_classes=num_classes)

    if pretrain:
        weight_root = "D:\weight"
        weights_dict = torch.load(weight_root+"/resnet34-333f7ec4.pth", map_location='cpu')

        if num_classes != 21:
            # 官方提供的预训练权重是21类(包括背景)
            for k in list(weights_dict.keys()):
                if "classifier.4" in k:
                    del weights_dict[k]

        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    return model


def main(args):
    #
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
#     device = torch.device("cpu")
    print(device)
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> train.txt
    # 加载训练集
    train_dataset = VOCSegmentation(args.data_path,
                                    year="2012",
                                    transforms=get_transform(train=True),
                                    txt="train.txt")
    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> val.txt
    # 加载测试数据
    val_dataset = VOCSegmentation(args.data_path,
                                  year="2012",
                                  transforms=get_transform(train=False),
                                  txt="val.txt")

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True,
                              pin_memory=True,
                              collate_fn=train_dataset.collate_fn)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            num_workers=num_workers,
                            pin_memory=True,
                            collate_fn=val_dataset.collate_fn)
    model = create_model(aux=args.aux, num_classes=num_classes)
    model.to(device)

    params_to_optimize = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model.classifier.parameters() if p.requires_grad]}
    ]

    if args.aux:
        params = [p for p in model.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        confmat = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n"
            f.write(train_info + val_info + "\n\n")
        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()
        torch.save(save_file, "save_weights/model_{}.pth".format(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    """
    这是训练参数
    :return:
    """

    parser = argparse.ArgumentParser(description="pytorch fcn training")
    # 设置数据集的位置
    parser.add_argument("--data-path", default= "D:\VOCtrainval_11-May-2012\VOCdevkit",
                        help="VOCdevkit")
    # 设置分割的种类,一共20种
    parser.add_argument("--num-classes", default=20, type=int)
    parser.add_argument("--aux", default=True, type=bool, help="auxilier loss")
    # 设置训练的设备
    parser.add_argument("--device", default="cuda", help="training device")
    # 设置一次训练的批次
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    # 设置训练的次数
    parser.add_argument("--epochs", default=30, type=int, metavar="N",
                        help="number of total epochs to train")
    # 设置学习率
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    # 设置
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')

    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args(args=[])

    return args


def main_process():
    freeze_support()
    args = parse_args()

    # 保存权重
    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)
    print("运行结束")


if __name__ == "__main__":
    main_process()
