import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, dataloader
import torchvision.transforms as transforms
from torchvision import datasets
import os
import numpy as np
def build_dataloader(batch_size, num_workers):
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    train_dataset =torchvision.datasets.ImageFolder(root='../../../../../liyc/data/imagenet2012/train',transform=train_transform)
    train_dataset_loader =DataLoader(train_dataset,batch_size=batch_size, shuffle=True,num_workers=num_workers, pin_memory=True)

    val_dataset = torchvision.datasets.ImageFolder(root='../../../../../liyc/data/imagenet2012/validation/',transform=valid_transform)
    val_dataset_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    # train_dataset = datasets.ImageNet(root='../../../../../liyc/data/imagenet2012/train', transform=train_transform)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
    #                                                num_workers=4, pin_memory=True, sampler=None)
    # # train_dataset_loader =DataLoader(train_dataset,batch_size=4, shuffle=True,num_workers=4
    # val_dataset = torchvision.datasets.ImageFolder(root='../../../../../liyc/data/imagenet2012/validation/',transform=valid_transform)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                                #  num_workers=4, pin_memory=True)
    return train_dataset, train_dataset_loader, val_dataset, val_dataset_loader
# val_loader, len = build_dataloader()
# print(len)
# for i, (input, target) in enumerate(val_loader):
#     # print(type(input), input.shape)
#     import pdb;pdb.set_trace()
# print(dataloader)

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, label, topk=(1,), step=0, epoch=0):
    maxk = max(topk)
    batch_size = label.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))
    # if (step == 200 or step == 400 or step == 600) and (epoch == 0):
    #     print(label, pred)
    #     print(label.view(1, -1).shape, pred.shape)
    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, iters, tag=''):
    if not os.path.exists("./snapshots"):
        os.makedirs("./snapshots")
    filename = os.path.join("./snapshots/{}_ckpt_{:04}.pth.tar".format(tag, iters))
    torch.save(state, filename)


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def load_model(model, path):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict, strict=True)
    return model

def load_to_MultiModel(MultiModel, path):
    pretrained_model = torch.load(path)
    # print("loaded")
    # test = MobileNetV3()
    model_dict = MultiModel.state_dict()
    state = {}
    for k, v in pretrained_model.items():
        if k in model_dict.keys():
            state[k] = v
            # print(k, "is in new model")
        else:
            if k[10] == '.':
                key = k[0:11] + '0.' + k[11:]
            else:
                key = k[0:12] + '0.' + k[12:]
            # print(key)
            state[key] = v
    model_dict.update(state)
    MultiModel.load_state_dict(model_dict)
    for (name, parameter) in MultiModel.named_parameters():
        if name in state:
            # print(";;;", name)
            parameter.requires_grad = False
        # print(k, v.shape)
        # if "features.0" in k:
        #     print(k)
        # print(type(k))
    # print("////////////////////////////////////////////////")
    # for k, v in pretrained_model.items():
    #     if k == "features.14.conv.0.weight":
    #         print(v)
    # for k, v in MultiModel.state_dict().items():
    #     if k == "features.14.0.conv.0.weight":
    #         print(v)
    return MultiModel
