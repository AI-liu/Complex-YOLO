import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np

from complexYOLO import ComplexYOLO
from kitti import KittiDataset
from region_loss import RegionLoss


batch_size=12

# dataset
dataset=KittiDataset(root='/home/yuliu/KITTI',set='train')
data_loader = data.DataLoader(dataset, batch_size, shuffle=True, pin_memory=False)

model = ComplexYOLO()
model.cuda()

# define optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-5 ,momentum = 0.9 , weight_decay = 0.0005)

# define loss function
region_loss = RegionLoss(num_classes=8, num_anchors=5)



for epoch in range(200):


   for group in optimizer.param_groups:
       if(epoch>=4 & epoch<80):
           group['lr'] = 1e-4
       if(epoch>=80 & epoch<160):
           group['lr'] = 1e-5
       if(epoch>=160):
           group['lr'] = 1e-6



   for batch_idx, (rgb_map, target) in enumerate(data_loader):          
          optimizer.zero_grad()

          rgb_map = rgb_map.view(rgb_map.data.size(0),rgb_map.data.size(3),rgb_map.data.size(1),rgb_map.data.size(2))
          output = model(rgb_map.float().cuda())

          loss = region_loss(output,target)
          loss.backward()
          optimizer.step()

   if (epoch % 10 == 0):
       torch.save(model, "ComplexYOLO_epoch"+str(epoch))
