import torch
from torch._C import parse_ir, set_flush_denormal
import torch.nn as nn
from transformers import RobertaModel
from utils.constant import CUDA


class ECIRoberta(nn.Module):
    def __init__(self, num_classes, dataset, mlp_size, roberta_type, finetune, loss=None, sub=True, mul=True):
        super().__init__()
        self.num_classes = num_classes
        self.data_set = dataset
        self.mlp_size = mlp_size
        self.robera = RobertaModel.from_pretrained(roberta_type)
        self.sub = sub
        self.mul = mul
        self.finetune = finetune
        if roberta_type == 'roberta-base':
            self.roberta_dim = 768
        if roberta_type == 'roberta-large':
            self.roberta_dim = 1024

        if dataset == "HiEve":
            weights = [993.0/333, 993.0/349, 933.0/128, 933.0/383]
        weights = torch.tensor(weights)
        if loss == None:
            self.loss = nn.CrossEntropyLoss(weight=weights)
        else:
            self.loss = loss

        if sub==True and mul==True:
            self.fc1 = nn.Linear(self.roberta_dim*4, self.mlp_size*2)
            self.fc2 = nn.Linear(self.mlp_size*2, num_classes)
        if (sub==True and  mul==False) or (sub==False and mul==True):
            self.fc1 = nn.Linear(self.roberta_dim*3, int(self.mlp_size*1.75))
            self.fc2 = nn.Linear(int(self.mlp_size*1.75), num_classes)
        if not (sub and mul):
            self.fc1 = nn.Linear(self.roberta_dim*2, int(self.mlp_size))
            self.fc2 = nn.Linear(int(self.mlp_size), num_classes)

        # print(self.fc1)

        self.relu = nn.LeakyReLU(0.2, True)
    
    def forward(self, x_sent, y_sent, x_position, y_position, xy):
        batch_size = x_sent.size(0)
        # print(x_sent.size())

        if self.finetune:
            output_x = self.robera(x_sent)[0]
            output_y = self.robera(y_sent)[0]
        else:
            with torch.no_grad():
                output_x = self.robera(x_sent)[0]
                output_y = self.robera(y_sent)[0]

        output_A = torch.cat([output_x[i, x_position[i], :].unsqueeze(0) for i in range(0, batch_size)])
        output_B = torch.cat([output_y[i, y_position[i], :].unsqueeze(0) for i in range(0, batch_size)])
        # print(output_B.size())
        if self.sub and self.mul:
            sub = torch.sub(output_A, output_B)
            mul = torch.mul(output_A, output_B)
            presentation = torch.cat([output_A, output_B, sub, mul], 1)
        if self.sub==True and self.mul==False:
            sub = torch.sub(output_A, output_B)
            presentation = torch.cat([output_A, output_B, sub], 1)
        if self.sub==False and self.mul==True:
            mul = torch.mul(output_A, output_B)
            presentation = torch.cat([output_A, output_B, mul], 1)
        
        # print(presentation.size())
        logits = self.fc2(self.relu(self.fc1(presentation)))
        loss = self.loss(logits, xy)
        return logits, loss