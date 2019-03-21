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
dataset=KittiDataset(root='/home/ai/KITTI',set='train')
data_loader = data.DataLoader(dataset, batch_size, shuffle=True)

model = ComplexYOLO()
model.cuda()

# define optimizer
optimizer = optim.Adam(model.parameters())

# define loss function
region_loss = RegionLoss(num_classes=8, num_anchors=5)



for epoch in range(200):

   for batch_idx, (rgb_map, target) in enumerate(data_loader):          
          optimizer.zero_grad()

          rgb_map = rgb_map.view(rgb_map.data.size(0),rgb_map.data.size(3),rgb_map.data.size(1),rgb_map.data.size(2))
          output = model(rgb_map.float().cuda())

          loss = region_loss(output,target)
          loss.backward()
          optimizer.step()

   if (epoch % 2 == 0):
       torch.save(model, "ComplexYOLO_epoch"+str(epoch))
