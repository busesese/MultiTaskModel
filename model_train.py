# -*- coding: utf-8 -*-
# @Time    : 2021-04-19 17:10
# @Author  : WenYi
# @Contact : wenyi@cvte.com
# @Description : model train function

import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

def train_model(model, train_loader, val_loader, epoch, loss_function, optimizer, path, early_stop):
    """
    pytorch model train function
    :param model: pytorch model
    :param train_loader: dataloader, train data loader
    :param val_loader: dataloader, val data loader
    :param epoch: int, number of iters
    :param loss_function: loss function of train model
    :param optimizer: pytorch optimizer
    :param path: save path
    :param early_stop: int, early stop number
    :return: None
    """
    # use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 多少步内验证集的loss没有变小就提前停止
    patience, eval_loss = 0, 0
    
    # train
    for i in range(epoch):
        y_train_income_true = []
        y_train_income_predict = []
        y_train_marry_true = []
        y_train_marry_predict = []
        total_loss, count = 0, 0
        for idx, (x, y1, y2) in tqdm(enumerate(train_loader), total=len(train_loader)):
            x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
            predict = model(x)
            y_train_income_true += list(y1.squeeze().cpu().numpy())
            y_train_marry_true += list(y2.squeeze().cpu().numpy())
            y_train_income_predict += list(predict[0].squeeze().cpu().detach().numpy())
            y_train_marry_predict += list(predict[1].squeeze().cpu().detach().numpy())
            loss_1 = loss_function(predict[0], y1.unsqueeze(1).float())
            loss_2 = loss_function(predict[1], y2.unsqueeze(1).float())
            loss = loss_1 + loss_2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
            count += 1
        torch.save(model, path.format(i + 1))
        income_auc = roc_auc_score(y_train_income_true, y_train_income_predict)
        marry_auc = roc_auc_score(y_train_marry_true, y_train_marry_predict)
        print("Epoch %d train loss is %.3f, income auc is %.3f and marry auc is %.3f" % (i + 1, total_loss / count,
                                                                                         income_auc, marry_auc))
        
        # 验证
        total_eval_loss = 0
        model.eval()
        count_eval = 0
        y_val_income_true = []
        y_val_marry_true = []
        y_val_income_predict = []
        y_val_marry_predict = []
        for idx, (x, y1, y2) in tqdm(enumerate(val_loader), total=len(val_loader)):
            x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
            predict = model(x)
            y_val_income_true += list(y1.squeeze().cpu().numpy())
            y_val_marry_true += list(y2.squeeze().cpu().numpy())
            y_val_income_predict += list(predict[0].squeeze().cpu().detach().numpy())
            y_val_marry_predict += list(predict[1].squeeze().cpu().detach().numpy())
            loss_1 = loss_function(predict[0], y1.unsqueeze(1).float())
            loss_2 = loss_function(predict[1], y2.unsqueeze(1).float())
            loss = loss_1 + loss_2
            total_eval_loss += float(loss)
            count_eval += 1
        income_auc = roc_auc_score(y_val_income_true, y_val_income_predict)
        marry_auc = roc_auc_score(y_val_marry_true, y_val_marry_predict)
        print("Epoch %d val loss is %.3f, income auc is %.3f and marry auc is %.3f" % (i + 1,
                                                                                       total_eval_loss / count_eval,
                                                                                       income_auc, marry_auc))
        
        # earl stopping
        if i == 0:
            eval_loss = total_eval_loss / count_eval
        else:
            if total_eval_loss / count_eval < eval_loss:
                eval_loss = total_eval_loss / count_eval
            else:
                if patience < early_stop:
                    patience += 1
                else:
                    print("val loss is not decrease in %d epoch and break training" % patience)
                    break
