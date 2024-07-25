import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from Helpers.Variables import device, FILENAME_RES, FILENAME_HIST, FILENAME_HISTSUM, METRICS
import random

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    
seed_everything()

class Trainer():
    def __init__(self, args, model, MODEL_PATH, res_dir):
        self.args = args
        self.model = model
        self.MODEL_PATH = MODEL_PATH
        
        self.set_optimizer()
        self.lossfn = nn.CrossEntropyLoss(reduction='mean')
        self.ranking_criterion = nn.MarginRankingLoss(margin=0.0).to(device)


    def set_optimizer(self):
        if self.args.optimizer=='Adam':
            adam = torch.optim.Adam(self.model.parameters(), lr=float(self.args.lr))
            self.optimizer = adam
        elif self.args.optimizer=='AdamW':
            adamw = torch.optim.AdamW(self.model.parameters(), lr=float(self.args.lr))
            self.optimizer = adamw
        if self.args.scheduler == 'StepLR': 
            self.scheduler = StepLR(self.optimizer, step_size=self.args.step_size, gamma=0.2)
        elif self.args.scheduler == 'CosineAnnealingLR':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10, eta_min=0)
        elif self.args.scheduler == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, min_lr=1e-4, patience=0, verbose=1) # original patience=0, F1 score -> max
    
    def train(self, train_loader, valid_loader):
        best_score = np.inf  # 최소 손실을 저장하는 변수, 초기값을 무한대로 설정
        history = pd.DataFrame()  # 성능 기록

        for epoch_idx in range(0, int(self.args.EPOCH)):
            result = self.train_epoch(train_loader, epoch_idx)  # train
            history = pd.concat([history, pd.DataFrame(result).T], axis=0, ignore_index=True)

            val_result = self.eval("valid", valid_loader, epoch_idx)
            valid_loss = val_result  # eval 함수에서 반환된 손실 값

            # 현재 손실이 이전 최소 손실보다 낮은 경우
            if valid_loss < best_score: 
                # 학습률 조정
                if self.args.scheduler == 'ReduceLROnPlateau':
                    self.scheduler.step(valid_loss)
                else: 
                    self.scheduler.step()
                
                print(f'Validation loss decreased ({best_score:.5f} --> {valid_loss:.5f}). Saving model ...')
                best_score = valid_loss  # 최소 손실 업데이트
                torch.save(self.model.state_dict(), self.MODEL_PATH)  # 모델 저장

            else:
                # 손실이 감소하지 않은 경우 학습률만 조정
                if self.args.scheduler == 'ReduceLROnPlateau':
                    self.scheduler.step(valid_loss)
                else: 
                    self.scheduler.step()

            print('')

        # Load best performance model
        self.model.load_state_dict(torch.load(self.MODEL_PATH))

        history.columns = METRICS
        return history     
       
    """ EVALUATE """
    def eval(self, phase, loader, epoch=0):
        self.model.eval() ## evaluation mode로 변경 (dropout 사용 중지)
        test_history = pd.DataFrame()
        test_loss = []
        preds=[]
        targets=[]

        with torch.no_grad(): # .eval함수와 torch.no_grad함수를 같이 사용하는 경향
            for datas in loader:
                data, target = datas['data'], datas['label'].squeeze()
                logit = self.model(data)[-1]

                test_loss.append(self.lossfn(logit, target).mean().item()) # sum up batch loss
                pred = logit.argmax(dim=1,keepdim=False)# get the index of the max probability
                target = target.argmax(dim=1,keepdim=False)
                
                preds.extend(pred.cpu().numpy())
                targets.extend(target.cpu().numpy())
        
        loss = sum(test_loss)/len(loader) # 평균 손실 계산 방식 수정
        acc=accuracy_score(targets, preds)
        bacc=balanced_accuracy_score(targets,preds)
        f1=f1_score(targets,preds, average='macro', zero_division=0)
        preci=precision_score(targets,preds, average='macro', zero_division=0)
        recall=recall_score(targets,preds, average='macro', zero_division=0)

        print('<{}> epoch {} -- Loss: {:.5f}, Accuracy: {:.5f}%, Balanced Accuracy: {:.5f}%, f1score: {:.5f}, precision: {:.5f}, recall: {:.5f}'
            .format(phase, epoch, loss, acc*100, bacc*100, f1, preci, recall))

        if phase == 'valid':    
            return loss 
        elif phase=="test":
            result = [loss, acc, bacc, f1, preci, recall]
            test_history = pd.concat([test_history, pd.DataFrame(result).T], axis=0, ignore_index=True)
            test_history.columns = METRICS
            return test_history
        
    """ Train """
    def train_epoch(self, train_loader, epoch=0):
        """
            x_train: (N, Ch, Seq)
            y_train: (N, classes)
            each element -> numpy array
        """
        self.model.train()      
        preds=[]
        targets=[]
        
        for datas in train_loader:
            data, target = datas['data'], datas['label'].squeeze()
            
            if self.args.data_type == 'Pilot_Fatigue':
                f14, f24, f34, f44, mid1, mid2, mid3, mid4, logit = self.model(data)
                # Calculate individual losses
                basic_loss = self.lossfn(logit, target)
                L1, L2, L3, L4 = F.cross_entropy(mid1, target.argmax(dim=1)), F.cross_entropy(mid2, target.argmax(dim=1)), F.cross_entropy(mid3, target.argmax(dim=1)), F.cross_entropy(mid4, target.argmax(dim=1))
                final_loss = basic_loss + f14.mean() * L1 + f24.mean() * L2 + f34.mean() * L3 + f44.mean() * L4

            elif self.args.data_type == 'MS' or self.args.data_type == 'manD':
                f15, f25, f35, f45, f55, mid1, mid2, mid3, mid4, mid5, logit = self.model(data)
                # Calculate individual losses
                basic_loss = self.lossfn(logit, target)
                L1, L2, L3, L4, L5 = F.cross_entropy(mid1, target.argmax(dim=1)), F.cross_entropy(mid2, target.argmax(dim=1)), F.cross_entropy(mid3, target.argmax(dim=1)), F.cross_entropy(mid4, target.argmax(dim=1)), F.cross_entropy(mid5, target.argmax(dim=1))
                final_loss = basic_loss + f15.mean() * L1 + f25.mean() * L2 + f35.mean() * L3 + f45.mean() * L4 + f55.mean() * L5
            
            else:   
                f13, f23, f33, mid1, mid2, mid3, logit = self.model(data)
                # Calculate individual losses
                basic_loss = self.lossfn(logit, target)
                L1, L2, L3 = F.cross_entropy(mid1, target.argmax(dim=1)), F.cross_entropy(mid2, target.argmax(dim=1)), F.cross_entropy(mid3, target.argmax(dim=1))
                final_loss = basic_loss + f13.mean() * L1 + f23.mean() * L2 + f33.mean() * L3

            final_loss = final_loss.mean() 

            self.optimizer.zero_grad()
            final_loss.backward()
            self.optimizer.step()

            pred = logit.argmax(dim=1)
            target = target.argmax(dim=1)
            
            preds.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())
            
        loss = final_loss.item()
        acc = accuracy_score(targets, preds)
        bacc=balanced_accuracy_score(targets, preds)
        f1=f1_score(targets, preds, average='macro')
        preci=precision_score(targets, preds, average='macro', zero_division=0)
        recall=recall_score(targets, preds, average='macro', zero_division=0)
        
        
        print('<train> epoch {} -- Loss: {:.5f}, Accuracy: {:.5f}%, Balanced Accuracy: {:.5f}%, f1score: {:.5f}, precision: {:.5f}, recall: {:.5f}'
            .format(epoch, loss, acc*100, bacc*100, f1, preci, recall))

        
        return [loss, acc, bacc, f1, preci, recall]

        
    def save_result(self, tr_history, ts_history, res_dir):
        # save test history to csv
        res_path = os.path.join(res_dir, FILENAME_RES)
        ts_history.to_csv(res_path)
        print('Evaluation result saved')
        
        # save train history to csv
        hist_path = os.path.join(res_dir, FILENAME_HIST)
        histsum_path = os.path.join(res_dir, FILENAME_HISTSUM)
        tr_history.to_csv(hist_path)
        tr_history.describe().to_csv(histsum_path)
        print('History & History summary result saved')
        print('Tensorboard ==> \"tensorboard --logdir=runs\" \n')



