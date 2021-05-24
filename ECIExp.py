from typing import Dict
from torch.utils.data.dataloader import DataLoader
import tqdm
from models.roberta_model_multi import ECIRobertaJointTask
import torch
import torch.nn as nn
import time
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report, accuracy_score
from utils.tools import CUDA, format_time
from os import path
import math
from utils.tools import *
import math


class EXP():
    def __init__(self, model:ECIRobertaJointTask, epochs, b_lr, m_lr, decay_rate, b_scheduler_lambda, m_lr_step,
                train_dataloader:DataLoader, validate_dataloaders: Dict, test_dataloaders: Dict, 
                best_path, weight_decay=0.01, train_lm_epoch=3, warmup_proportion=0.1) -> None:
        self.model = model

        self.epochs = epochs

        self.train_dataloader = train_dataloader
        self.test_datatloaders = list(test_dataloaders.values())
        self.validate_dataloaders = list(validate_dataloaders.values())
        self.datasets = list(test_dataloaders.keys())
        
        self.decay_rate = decay_rate
        self.b_lr = b_lr
        self.mlp_lr = m_lr
        self.bert_param_list = []
        self.mlp_param_list = []
        self.train_roberta_epoch = train_lm_epoch
        self.warmup_proportion = warmup_proportion
        
        mlp = ['fc1', 'fc2', 'lstm', 'pos_emb']
        no_decay = ['bias', 'gamma', 'beta']
        group1=['layer.0.','layer.1.','layer.2.','layer.3.']
        group2=['layer.4.','layer.5.','layer.6.','layer.7.']
        group3=['layer.8.','layer.9.','layer.10.','layer.11.']
        group_all = group1 + group2 + group3 
        
        self.b_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in mlp) and not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay_rate': 0.01, 'lr': self.b_lr}, # all params not include bert layers 
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in mlp) and not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay_rate': 0.01, 'lr': self.b_lr*(self.decay_rate**2)}, # param in group1
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in mlp) and not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay_rate': 0.01, 'lr': self.b_lr*(self.decay_rate**1)}, # param in group2
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in mlp) and not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay_rate': 0.01, 'lr': self.b_lr*(self.decay_rate**0)}, # param in group3
            # no_decay
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in mlp) and any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay_rate': 0.00, 'lr': self.b_lr}, # all params not include bert layers 
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in mlp) and any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay_rate': 0.00, 'lr': self.b_lr*(self.decay_rate**2)}, # param in group1
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in mlp) and any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay_rate': 0.00, 'lr': self.b_lr*(self.decay_rate**1)}, # param in group2
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in mlp) and any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay_rate': 0.00, 'lr': self.b_lr*(self.decay_rate**0)}, # param in group3
        ]
        self.mlp_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in mlp) and not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01, 'lr': self.mlp_lr},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in mlp) and any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.00, 'lr': self.mlp_lr},
            ]
        optimizer_parameters = self.b_parameters + self.mlp_parameters 
        self.optimizer = optim.AdamW(optimizer_parameters, amsgrad=True, weight_decay=weight_decay)

        self.num_training_steps = len(self.train_dataloader) * self.train_roberta_epoch
        self.num_warmup_steps = int(self.warmup_proportion * self.num_training_steps)
        
        def linear_lr_lambda(current_step: int):
            if current_step < self.num_warmup_steps:
                return float(current_step) / float(max(1, self.num_warmup_steps))
            if current_step >= self.num_training_steps:
                return 0
            return max(
                0.0, float(self.num_training_steps - current_step) / float(max(1, self.num_training_steps - self.num_warmup_steps))
            )
        
        def cosin_lr_lambda(current_step: int):
            if current_step < self.num_warmup_steps:
                return float(current_step) / float(max(1, self.num_warmup_steps))
            progress = float(current_step - self.num_warmup_steps) / float(max(1, self.num_training_steps - self.num_warmup_steps))
            if current_step >= self.num_training_steps:
                return 0.0
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(0.5) * 2.0 * progress)))
        
        def m_lr_lambda(current_step: int):
            return 0.5 ** int(current_step / (m_lr_step*len(self.train_dataloader)))
        
        if b_scheduler_lambda == "cosin":
            b_lambda = cosin_lr_lambda
        elif b_scheduler_lambda == "linear":
            b_lambda = linear_lr_lambda
        else:
            print("We haven't support that lambda!!")
        lamd = [b_lambda] * 8
        mlp_lambda = [m_lr_lambda] * 2
        lamd.extend(mlp_lambda)
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lamd)

        self.best_micro_f1 = [-0.1]*len(self.test_datatloaders)
        self.sum_f1 = -0.1
        self.best_cm = [None]*len(self.test_datatloaders)
        self.best_path = best_path
    
    def train(self):
        try:
            total_t0 = time.time()
            for i in range(0, self.epochs):
                if i >= self.train_roberta_epoch:
                    for group in self.b_parameters:
                        for param in group['params']:
                            param.requires_grad = False

                print("")
                print('======== Epoch {:} / {:} ========'.format(i + 1, self.epochs))

                t0 = time.time()
                self.model.train()
                self.model.zero_grad()
                self.train_loss = 0.0
                for step, batch in tqdm.tqdm(enumerate(self.train_dataloader), desc="Training process", total=len(self.train_dataloader)):
                    x_sent, y_sent, x_position, y_position, x_sent_pos, y_sent_pos, flag, xy = batch[2:]
                    if CUDA:
                        x_sent = x_sent.cuda()
                        y_sent = y_sent.cuda()
                        x_position = x_position.cuda()
                        y_position = y_position.cuda()
                        xy = xy.cuda()
                        flag = flag.cuda()
                        x_sent_pos = x_sent_pos.cuda() 
                        y_sent_pos = y_sent_pos.cuda()

                    logits, loss = self.model(x_sent, y_sent, x_position, y_position, xy, flag, x_sent_pos, y_sent_pos)
                    self.train_loss += loss.item()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    # if step%50==0 and not step==0:
                    #     elapsed = format_time(time.time() - t0)
                    #     print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(self.train_dataloader), elapsed))
                    #     print("LR: {} - {}".format(self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[-1]['lr']))
                
                epoch_training_time = format_time(time.time() - t0)
                print("  Total training loss: {0:.2f}".format(self.train_loss))
                self.evaluate()
                
            print("Training complete!")
            print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
            print("Best micro F1:{}".format(self.best_micro_f1))
            print("Best confusion matrix: ")
            for cm in self.best_cm:
                print(cm)
            return self.best_micro_f1, self.best_cm, self.sum_f1
        except KeyboardInterrupt:
            return self.best_micro_f1, self.best_cm, self.sum_f1

    def evaluate(self, is_test=False):
        t0 = time.time()
        F1s = []
        sum_f1 = 0.0
        best_cm = []
        for i in range(0, len(self.test_datatloaders)):
            dataset = self.datasets[i]
            print("---------------------{}----------------------".format(dataset))
            if is_test:
                dataloader = self.test_datatloaders[i]
                self.model = torch.load(self.best_path)
                print("Loaded best model for test!")
                print("Running on testset......")
            else:
                dataloader = self.validate_dataloaders[i]
                print("Running on dev set......")
            
            self.model.eval()
            pred = []
            gold = []
            for batch in tqdm.tqdm(dataloader, desc="Process"):
                x_sent, y_sent, x_position, y_position, x_sent_pos, y_sent_pos, flag, xy = batch[2:]
                if CUDA:
                    x_sent = x_sent.cuda()
                    y_sent = y_sent.cuda()
                    x_position = x_position.cuda()
                    y_position = y_position.cuda()
                    xy = xy.cuda()
                    flag = flag.cuda()
                    x_sent_pos = x_sent_pos.cuda() 
                    y_sent_pos = y_sent_pos.cuda()
                logits, loss = self.model(x_sent, y_sent, x_position, y_position, xy, flag, x_sent_pos, y_sent_pos)

                label_ids = xy.cpu().numpy()
                y_pred = torch.max(logits, 1).indices.cpu().numpy()
                pred.extend(y_pred)
                gold.extend(label_ids)

            P, R, F1 = precision_recall_fscore_support(gold, pred, average="micro")[0:3]
            CM = confusion_matrix(gold, pred)
            print("  P: {0:.3f}".format(P))
            print("  R: {0:.3f}".format(R))
            print("  F1: {0:.3f}".format(F1))
            print("Classification report: \n {}".format(classification_report(gold, pred)))
            
            if is_test:
                print("Test result:")
                print("  P: {0:.3f}".format(P))
                print("  R: {0:.3f}".format(R))
                print("  F1: {0:.3f}".format(F1))
                print("  Confusion Matrix")
                print(CM)
                print("Classification report: \n {}".format(classification_report(gold, pred)))
            
            sum_f1 += F1
            best_cm.append(CM)
            F1s.append(F1)

        if is_test == False:
            if sum_f1 > self.sum_f1 or path.exists(self.best_path) == False:
                self.sum_f1 = sum_f1
                self.best_cm = best_cm
                self.best_micro_f1 = F1s
                torch.save(self.model, self.best_path)
        return F1s
