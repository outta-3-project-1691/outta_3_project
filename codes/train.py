import torch
import torch.nn as nn
import torch.nn.functional as F

from .read_dataset import *
from .preprop import *
from .model import segmentation_model

class SCLoss(nn.Module):
    def __init__(self):
        super(SCLoss, self).__init__()
    
    def forward(self, pred, target):
        # Pred와 Target은 (B, 1, H, W)의 형태라고 가정
        # 그레이디언트 차이를 계산하기 위해 x와 y 방향으로 편미분
        pred_dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        pred_dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        
        target_dx = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_dy = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        
        # x와 y 방향의 그레이디언트 차이를 합산
        loss_x = F.l1_loss(pred_dx, target_dx)
        loss_y = F.l1_loss(pred_dy, target_dy)
        
        return loss_x + loss_y
def epoch_loop(model, criterion, sc_loss, L, optimizer, dataloader, valid=False):
    epoch_loss = 0

    model.train()

    for inputs, targets in tqdm(dataloader, leave=False):
        inputs, targets = inputs.to(device=model.device), targets.to(device=model.device)
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
    train_imgs = torch.cat(img_list, dim=0)
    train_gts = torch.cat(gt_list, dim=0)

    # TensorDataset으로 변환
    train_dataset = TensorDataset(train_imgs, train_gts)
    
    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
  
    model.train()

    for i in tqdm(range(epoch)):
      epoch_loop(model, criterion, sc_loss, L,optimizer, train_loader)
      
      torch.save(model.state_dict(), f'model_state_dict_epoch_{i}.pth')
    optimizer = Adam(model.parameters(), lr=1e-8)

    train(train_loader, model, epoch, optimizer, criterion)