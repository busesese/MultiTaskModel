# -*- coding: utf-8 -*-
# @Time    : 2021-04-19 17:25
# @Author  : WenYi
# @Contact : wenyi@cvte.com
# @Description :  script description


from utils import data_preparation, TrainDataSet
from torch.utils.data import DataLoader
from model_train import train_model
from esmm import ESMM
from mmoe import MMOE
import torch
import torch.nn as nn


def main():
    train_data, test_data, user_feature_dict, item_feature_dict = data_preparation()
    train_dataset = (train_data.iloc[:, :-2].values, train_data.iloc[:, -2].values, train_data.iloc[:, -1].values)
    # val_dataset = (val_data.iloc[:, :-2].values, val_data.iloc[:, -2].values, val_data.iloc[:, -1].values)
    test_dataset = (test_data.iloc[:, :-2].values, test_data.iloc[:, -2].values, test_data.iloc[:, -1].values)
    train_dataset = TrainDataSet(train_dataset)
    # val_dataset = TrainDataSet(val_dataset)
    test_dataset = TrainDataSet(test_dataset)

    # dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # pytorch优化参数
    learn_rate = 0.01
    bce_loss = nn.BCEWithLogitsLoss()
    early_stop = 3
    
    # train model
    # esmm Epoch 17 val loss is 1.164, income auc is 0.875 and marry auc is 0.953
    esmm = ESMM(user_feature_dict, item_feature_dict, emb_dim=64)
    optimizer = torch.optim.Adam(esmm.parameters(), lr=learn_rate)
    train_model(esmm, train_dataloader, test_dataloader, 20, bce_loss, optimizer, 'model/model_esmm_{}', early_stop)
    
    # mmoe
    mmoe = MMOE(user_feature_dict, item_feature_dict, emb_dim=64)
    optimizer = torch.optim.Adam(mmoe.parameters(), lr=learn_rate)
    train_model(mmoe, train_dataloader, test_dataloader, 20, bce_loss, optimizer, 'model/model_mmoe_{}', early_stop)
    

if __name__ == "__main__":
    main()