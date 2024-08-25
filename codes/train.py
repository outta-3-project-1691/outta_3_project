import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from tqdm.auto import tqdm

from .read_dataset import *
from .preprop import *
from .model import segmentation_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SCLoss(nn.Module):
    def __init__(self):
        super(SCLoss, self).__init__()
    
    def forward(self, pred, target):
        # Pred와 Target은 (B, 1, H, W)
        pred_dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        pred_dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        
        target_dx = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_dy = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        
        loss_x = F.l1_loss(pred_dx, target_dx)
        loss_y = F.l1_loss(pred_dy, target_dy)
        
        return loss_x + loss_y

class SigmoidLRScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epochs, min_lr, max_lr, last_epoch=-1):
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.max_lr = max_lr
        super(SigmoidLRScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        # 현재 진행된 epoch 비율
        epoch_ratio = self.last_epoch / self.total_epochs
        
        # 시그모이드 함수를 통해 학습률 계산
        sigmoid_factor = 1 / (1 + math.exp(-10 * (epoch_ratio - 0.5)))  # 시그모이드 함수
        new_lr = self.min_lr + (self.max_lr - self.min_lr) * sigmoid_factor
        
        return [new_lr for _ in self.optimizer.param_groups]

def epoch_loop(model, criterion, sc_loss, L, optimizer, dataloader, valid=False):
    epoch_loss = 0
    model.train()
    for inputs, targets in tqdm(dataloader, leave=False):
        inputs, targets = inputs.to(device=device), targets.to(device=device)
        targets = targets.long()

        # Forward pass
        outputs = model(inputs)

        # Main Loss 계산 (CrossEntropyLoss)
        ce_loss = criterion(outputs, targets)

        # SCLoss 계산
        outputs_probs = F.softmax(outputs, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=outputs.shape[1]).permute(0, 3, 1, 2).float()
        sc_loss_value = sc_loss(outputs_probs, targets_onehot)

        loss = ce_loss + L * sc_loss_value

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        del inputs, targets, outputs, ce_loss, outputs_probs, targets_onehot, sc_loss_value, loss
        torch.cuda.empty_cache()

    return epoch_loss/len(dataloader)


def train(img_list, gt_list, model, epoch, learning_rate, optimizer, criterion, data_len):
    # 리스트를 텐서로 결합
    L = 0.5
    sc_loss = SCLoss()
    scheduler = SigmoidLRScheduler(optimizer,epoch, learning_rate, 1e-8 )
    train_imgs = torch.tensor(np.array(img_list))
    train_gts = torch.tensor(np.array(gt_list)).squeeze(1)
    print()
    # TensorDataset으로 변환
    train_dataset = TensorDataset(train_imgs, train_gts)
    
    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
  
    model.train()

    for i in tqdm(range(epoch)):
      epoch_loop(model, criterion, sc_loss, L, optimizer, train_loader)
      
      print(scheduler.get_last_lr())
      scheduler.step()

      torch.save(model.state_dict(), f'model_state_dict_epoch_{i}.pth')
