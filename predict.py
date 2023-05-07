from collections import OrderedDict
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from operation import predict
from path import *
import torch
from dataset import RsDataset
from networks.USSFCNet import USSFCNet

print('CUDA: ', torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1]

src_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize(512),
    transforms.Normalize([0.5], [0.5])
])

label_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize(512),
])

model = USSFCNet(3, 1, ratio=0.5).to(device)
# model = torch.nn.DataParallel(model, device_ids=device_ids).to(device)
model_path = 'ckps/USSFCNet_LEVIRCD/best91.04_levir.pth'
ckps = torch.load(model_path, map_location='cuda:0')
model.load_state_dict(ckps)

dataset_test = RsDataset(test_src_t1, test_src_t2, test_label, test=True,
                         t1_transform=src_transform,
                         t2_transform=src_transform,
                         label_transform=label_transform)
dataloader_test = DataLoader(dataset_test,
                             batch_size=1,
                             shuffle=False,
                             num_workers=4)
pre_test, rec_test, f1_test, iou_test, kc_test = predict(model, dataloader_test)
print('test Pre:(%f,%f) test Recall:(%f,%f) test MeanF1Score:(%f,%f) test IoU:(%f,%f) test KC: %f' % (
    pre_test['precision_0'], pre_test['precision_1'], rec_test['recall_0'], rec_test['recall_1'], f1_test['f1_0'],
    f1_test['f1_1'], iou_test['iou_0'], iou_test['iou_1'], kc_test))
