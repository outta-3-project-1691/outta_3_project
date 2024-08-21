import torch
from torch.utils.data import DataLoader, TensorDataset
import h5py

def load_testset(file_path, train_len, valid_len, batch_size=32, num_workers=0):
    '''
    지정된 train과 valid 셋을 반환
    train_len+valid_len <= 10000 이도록 설정해야함

    return: train_loader, valid_loader
    '''
    
    # Train 및 Valid 데이터를 저장할 텐서 초기화
    train_imgs = []
    train_gts = []
    valid_imgs = []
    valid_gts = []
    
    with h5py.File(file_path, 'r') as file:
        # 각 데이터셋 로드
        data_img = file['ih']
        data_gt = file['b_']
        
        # 길이만큼 데이터 불러오기
        for i in range(train_len + valid_len):
            img = torch.tensor(data_img[i])  # NumPy 배열을 PyTorch 텐서로 변환
            gt = torch.tensor(data_gt[i])
            
            if i < train_len:
                train_imgs.append(img.unsqueeze(0))  # 차원 추가 후 리스트에 저장
                train_gts.append(gt)
            else:
                valid_imgs.append(img.unsqueeze(0))  # 차원 추가 후 리스트에 저장
                valid_gts.append(gt)
    
    # 리스트를 텐서로 결합
    train_imgs = torch.cat(train_imgs, dim=0)
    train_gts = torch.cat(train_gts, dim=0)
    valid_imgs = torch.cat(valid_imgs, dim=0)
    valid_gts = torch.cat(valid_gts, dim=0)
    
    # TensorDataset으로 변환
    train_dataset = TensorDataset(train_imgs, train_gts)
    valid_dataset = TensorDataset(valid_imgs, valid_gts)
    
    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, valid_loader

if __name__ == '__main__':
    file_path = 'dataset/G2_train.h5'
    train_len = 8000
    valid_len = 2000
    train_loader, valid_loader = load_testset(file_path, train_len, valid_len, batch_size=32, num_workers=0)