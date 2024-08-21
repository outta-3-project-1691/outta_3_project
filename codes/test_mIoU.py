import torch
import numpy as np

# preds, labels -> numpy array
def calculate_miou(preds, labels, num_classes=7):

    ious = []
    for i in range(num_classes):

        tp = np.sum((preds == i) & (labels == i))
        fp = np.sum((preds == i) & (labels != i))
        fn = np.sum((preds != i) & (labels == i))

        # IoU = Intersection (TP) / Union (TP + FP + FN)
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        ious.append(iou)
    
    return np.mean(ious)

def evaluate_model(img_list, gt_list, model, num_classes=7):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in zip(img_list, gt_list):

            inputs = inputs.to(model.device)
            labels = labels.to(model.device)
            
            # output -> (batch_size, num_classes, height, width)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # IoU calculation을 위한 변환
            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return calculate_miou(all_preds, all_labels, num_classes)
