import os
import torch.utils.data as data
from PIL import Image


class VOCSegmentation(data.Dataset):
    """
    数据集准备
    """
    def __init__(self, voc_root, year="2012", transforms=None, txt: str = "train.txt"):
        """
        :param voc_root: 数据集的路径
        :param year: 年份，现在只训练了2012年的voc数据集
        :param transforms: 是否对图片进行剪裁，默认不进行裁剪
        :param txt:
        """
        super(VOCSegmentation, self).__init__()
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        # 拼接路径
        root = os.path.join(voc_root, f"VOC{year}")
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        image_dir = os.path.join(root, 'JPEGImages')
        # 掩膜的路径位置
        mask_dir = os.path.join(root, 'SegmentationClass')
        txt_path = os.path.join(root, "ImageSets", "Segmentation", txt)

        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        # 根据Segmentation 文件夹下所以提供的train.txt,来进行图片的加载
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        # 掩膜图片位置
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))
        self.transforms = transforms

    def __getitem__(self, index):
        """

        :param index:
        :return: 返回原始图片信息,一个是原始图片，一个是分割好的图片信息
        """

        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        """
        对一整个batch 进行合并
        """
        images, targets = list(zip(*batch))
        batched_imgs = fill_list(images, fill_value=0)
        batched_targets = fill_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def fill_list(images, fill_value=0):
    """
    对一批图像进行批处理，使它们具有相同的形状
    :param images: 图片信息
    :param fill_value:
    :return:
    """

    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs
