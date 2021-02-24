import os
import pandas as pd
import time
import numpy as np

import torch
import torchvision
import torch.nn as nn

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import argparse
from ZXX_utils.load_csv_data import csv_Dataset
from ZXX_utils.IR_train_MLCL import train_2, test_3
from torch.nn import functional as F
import torch.nn.utils.weight_norm as weightNorm
from model_file.resnet import resnet50

parser = argparse.ArgumentParser(description='Using ShiftedSoftmax to FGOR')

parser.add_argument('--base_lr', dest='base_lr', type=float, default=5e-3)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=56)
parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-6)
parser.add_argument('--epoches', dest='epoches', type=int, default=100)
args = parser.parse_args()
print(args)
os.makedirs('Softmax', exist_ok=True)

train_csv_path = '../data/CUB_200_2011/train.csv'
test_csv_path = '../data/CUB_200_2011/test.csv'
recall_k = [1, 2, 4, 8, 16, 32]
precision_k = [100]
mAP_k = [100]

gamma = 0.1
sigma = 100
Lambda = 0.1

args.save_result_path = 'Softmax/trained'
args.model = os.path.join(args.save_result_path, 'best.pth')
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

class L2Norm(torch.nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        if len(norm.size()) == 1:
            x = x / norm.unsqueeze(-1).expand_as(x)
        else:
            [bs, ch, h, w] = x.size()
            norm = norm.view(bs, 1, h, w)
            x = x / norm.expand_as(x)
        return x

class cal_L2Norm(torch.nn.Module):
    def __init__(self):
        super(cal_L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)

        return norm

class ResNet50_IR(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        Resnet50featuremap = resnet50(pretrained=True)
        layer_conv1_conv4 = [Resnet50featuremap.conv1,
                             Resnet50featuremap.bn1,
                             Resnet50featuremap.relu,
                             Resnet50featuremap.maxpool,
                             ]

        for i in range(4):
            name = 'layer%d' % (i + 1)
            layer_conv1_conv4.append(getattr(Resnet50featuremap, name))

        conv1_conv5 = torch.nn.Sequential(*layer_conv1_conv4)
        self.features = conv1_conv5
        self.max_pool = torch.nn.AdaptiveMaxPool2d([1, 1])
        self.avg_pool = torch.nn.AdaptiveAvgPool2d([1, 1])
        self.alpha = sigma
        self.fc = weightNorm(nn.Linear(2048 * 2, 100, bias=False))

    def forward(self, x, train_flag=False):
        batchsize = x.size(0)
        x = self.features(x)
        
        if train_flag:
            mx = self.max_pool(x).view(batchsize, -1)
            ax = self.avg_pool(x).view(batchsize, -1)
            x = torch.cat([mx, ax], 1)
            norm = cal_L2Norm()(x)

            v = self.fc.weight_v
            g = self.fc.weight_g
            v = v / g
            x_ = self.fc(x) * self.alpha / norm.unsqueeze(1)
            return x_, v, x

        else:

            mx = self.max_pool(x).view(batchsize, -1)
            ax = self.avg_pool(x).view(batchsize, -1)
            x = torch.cat([mx, ax], 1)
            x = L2Norm()(x)
            return x

def main():
    print('Come to the main function')

    print(args)

    lr = args.base_lr
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    epoches = args.epoches
    model_path = args.model
    save_model_path = args.save_result_path

    train_data_list = pd.read_csv(train_csv_path, encoding='gbk')
    test_data_list = pd.read_csv(test_csv_path, encoding='gbk')

    model = ResNet50_IR()
    model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    milestones = [100, 200]
    model_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma= 0.1)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    imagesize = 280
    train_data = csv_Dataset(train_data_list,
                             transform=transforms.Compose([
                                 transforms.Resize(imagesize),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomCrop(imagesize),
                                 transforms.ToTensor(),
                                 normalize,
                             ]))
    test_data = csv_Dataset(test_data_list,
                            transform=transforms.Compose([
                                transforms.Resize(imagesize),
                                transforms.CenterCrop(imagesize),
                                transforms.ToTensor(),
                                normalize,
                            ]))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size= 20, shuffle=False, pin_memory=False, num_workers=4)

    best_acc = 0.0
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)

    if not os.path.exists(os.path.join(save_model_path, 'result')):
        os.mkdir(os.path.join(save_model_path, 'result'))

    recall_k_output, precision_k_output, mAP_k_output, np_predict = test_3(model, test_loader=test_loader,
                                                                           recall=recall_k,
                                                                           precision=precision_k,
                                                                           mAP=mAP_k)
    output_string = '%d epoch: ' % -1
    for k in recall_k:
        output_string = output_string + 'top%d:%4.2f, ' % (k, recall_k_output[str(k)] )
    for k in precision_k:
        output_string = output_string + "Precision_%d: %4.2f, " % (k, precision_k_output[str(k)])
    for k in mAP_k:
        output_string = output_string + "mAP_%d: %4.2f" % (k, mAP_k_output[str(k)])

    print(output_string)

    for epoch in range(epoches):
        train_acc, train_loss = train_2(model=model, train_loader=train_loader, criterion=criterion, optimizer=optimizer, Lambda=Lambda)
        print('%d\t%4.3f\t\t%4.2f%%\t' % (epoch, train_loss, train_acc))
        recall_k_output, precision_k_output, mAP_k_output, np_predict = test_3(model, test_loader=test_loader, recall=recall_k,
                                                                           precision=precision_k,
                                                                           mAP=mAP_k)

        model_scheduler.step()

        if recall_k_output['1'] > best_acc:
            best_acc = recall_k_output['1']
            print('*', end='')

        output_string = '%d epoch: ' % epoch
        for k in recall_k:
            output_string = output_string + 'top%d:%4.2f, ' % (k, recall_k_output[str(k)] )
        for k in precision_k:
            output_string = output_string + "Precision_%d: %4.2f, " % (k, precision_k_output[str(k)])
        for k in mAP_k:
            output_string = output_string + "mAP_%d: %4.2f" % (k, mAP_k_output[str(k)])

        print(output_string)


if __name__ == '__main__':
    main()








