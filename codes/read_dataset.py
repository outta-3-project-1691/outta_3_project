import numpy as np
import h5py

def read_dataset(file_path, data_len):
    img_list = []
    gt_list = []
    
    with h5py.File(file_path, 'r') as file:
        # 각 데이터셋 로드
        data_img = file['ih']
        data_gt = file['b_']
        
        # 길이만큼 데이터 불러오기
        for i in range(data_len):
            img_list.append(np.array(data_img[i]))
            gt_list.append(np.array(data_gt[i]))
    
    return img_list, gt_list

def read_dataset_train_valid(file_path, train_len, valid_len):
    '''
    지정된 train과 valid 셋을 반환
    train_len+valid_len <= 10000 이도록 설정해야함

    return:

    '''
    train_img_list = []
    train_gt_list = []
    valid_img_list = []
    valid_gt_list = []
    
    with h5py.File(file_path, 'r') as file:
        # 각 데이터셋 로드
        data_img = file['ih']
        data_gt = file['b_']
        
        # 길이만큼 데이터 불러오기
        for i in range(train_len+valid_len):
            if i < train_len:
                train_img_list.append(np.array(data_img[i]))
                train_gt_list.append(np.array(data_gt[i]))
            else:
                valid_img_list.append(np.array(data_img[i]))
                valid_gt_list.append(np.array(data_gt[i]))
        
    return train_img_list, train_gt_list, valid_img_list , valid_gt_list