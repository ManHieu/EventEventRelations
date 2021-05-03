from numpy.lib.function_base import average
from torch.utils.data.dataloader import DataLoader
from models.roberta_model import ECIRoberta
import torch
import torch.nn as nn
import time
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report, accuracy_score
from utils.tools import CUDA, format_time, metric
from os import path
from utils.tools import *

class EXP():
    def __init__(self, model:ECIRoberta, epochs, lr, train_dataloader:DataLoader, validate_dataloader:DataLoader, test_dataloader:DataLoader, best_path) -> None:
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.train_dataloader = train_dataloader
        self.test_datatloader = test_dataloader
        self.validate_dataloader = validate_dataloader

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, amsgrad=True)
        # self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.lr)
        self.best_micro_f1 = -0.1
        self.best_cm = []
        self.best_path = best_path
    
    def train(self):
        total_t0 = time.time()
        pre_F1 = 0.0
        pre_loss = 0.0
        for i in range(0, self.epochs):
            print("")
            print('======== Epoch {:} / {:} ========'.format(i + 1, self.epochs))
            print('Training...')

            t0 = time.time()
            self.model.train()
            self.train_loss = 0.0
            pre_F1 = 0.0
            for step, batch in enumerate(self.train_dataloader):
                if step%50==0 and not step==0:
                    elapsed = format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(self.train_dataloader), elapsed))
                x_sent, y_sent, x_position, y_position, x_sent_pos, y_sent_pos, xy = batch[2:]
                if CUDA:
                    x_sent = x_sent.cuda()
                    y_sent = y_sent.cuda()
                    x_position = x_position.cuda()
                    y_position = y_position.cuda()
                    xy = xy.cuda()
                logits, loss = self.model(x_sent, y_sent, x_position, y_position, xy)
                self.train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            
            epoch_training_time = format_time(time.time() - t0)
            print("  Total training loss: {0:.2f}".format(self.train_loss))
            print("  Training epoch took: {:}".format(epoch_training_time))
            current_F1 = self.evaluate()
            current_loss = self.train_loss
            if i%3 == 1:
                if abs(current_F1 - pre_F1) < 0.005 or (current_loss - pre_loss) > 500 or abs(current_loss - pre_loss) < 5:
                    break
            pre_loss = current_loss
            pre_F1 = current_F1
        
        print("Training complete!")
        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
        print("Best micro F1:{}".format(self.best_micro_f1))
        print("Bset confusion matrix: \n {} \n".format(self.best_cm))
        return self.best_micro_f1

    def evaluate(self, is_test=False):
        t0 = time.time()
        if is_test:
            dataloader = self.test_datatloader
            self.model = torch.load(self.best_path)
            print("Loaded best model for test!")
            print("Running on testset......")
        else:
            dataloader = self.validate_dataloader
            print("Running on dev set......")
        
        self.model.eval()
        pred = []
        gold = []
        for batch in dataloader:
            x_sent, y_sent, x_position, y_position, x_sent_pos, y_sent_pos, xy = batch[2:]
            if CUDA:
                x_sent = x_sent.cuda()
                y_sent = y_sent.cuda()
                x_position = x_position.cuda()
                y_position = y_position.cuda()
                xy = xy.cuda()
            logits, loss = self.model(x_sent, y_sent, x_position, y_position, xy)

            label_ids = xy.cpu().numpy()
            y_pred = torch.max(logits, 1).indices.cpu().numpy()
            pred.extend(y_pred)
            gold.extend(label_ids)

        validation_time = format_time(time.time() - t0)
        print("Eval took: {:}".format(validation_time))
        # print(len(pred))
        # print(len(gold))
        # Acc, P, R, F1, CM = metric(gold, y_pred)
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
        if is_test == False:
            if F1 > self.best_micro_f1 or path.exists(self.best_path) == False:
                self.best_micro_f1 = F1
                self.best_cm = CM
                torch.save(self.model, self.best_path)
        
        return F1
