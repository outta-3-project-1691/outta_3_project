import torch

# preds, labels -> torch.Tensor
def calculate_miou(preds, labels, num_classes=7):

    preds = preds.to('cuda')
    labels = labels.to('cuda')

    ious = []
    for i in range(num_classes):
        tp = torch.sum((preds == i) & (labels == i)).float()
        fp = torch.sum((preds == i) & (labels != i)).float()
        fn = torch.sum((preds != i) & (labels == i)).float()

        # IoU = Intersection (TP) / Union (TP + FP + FN)
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else torch.tensor(0.0).to('cuda')
        ious.append(iou.item())
    
    return torch.mean(torch.tensor(ious)).item()

def evaluate_model(img_list, gt_list, model, num_classes=7):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in zip(img_list, gt_list):

            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            
            # output -> (batch_size, num_classes, height, width)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.append(preds)
            all_labels.append(labels)
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return calculate_miou(all_preds, all_labels, num_classes)
