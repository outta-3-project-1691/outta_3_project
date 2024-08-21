import torch
import torchmetrics
from torch.utils.data import DataLoader

# F1 Score (또는 Dice Score) 평가 함수
def evaluate_model(model, dataloader, device):
    model.eval()
    f1_metric = torchmetrics.classification.BinaryF1Score().to(device)
    
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            # 이진화 (0.5 기준으로 binary mask 생성)
            preds = (outputs > 0.5).float()
            
            # F1 Score 업데이트
            f1_metric.update(preds, masks)
    
    # 최종 F1 Score 계산
    f1_score = f1_metric.compute().item()
    return f1_score