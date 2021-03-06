import torch
import torch.nn as nn
from transformers import RobertaModel
from models.losses.all_loss_aug import transitivity_loss_H_, transitivity_loss_T_, cross_category_loss_
from utils.tools import CUDA


HierPC = 1802.0
HierCP = 1846.0
HierCo = 758.0
HierNo = 63755.0 
HierTo = HierPC + HierCP + HierCo + HierNo # total number of event pairs
hier_weights = [0.25*HierTo/HierPC, 0.25*HierTo/HierCP, 0.25*HierTo/HierCo, 0.25*HierTo/HierNo]
temp_weights = [0.25*818.0/412.0, 0.25*818.0/263.0, 0.25*818.0/30.0, 0.25*818.0/13.0]

# roberta + MLP
class roberta_mlp(nn.Module):
    def __init__(self, num_classes, dataset, add_loss, lambdas, Sub = True, Mul = True, freq = None):
        super(roberta_mlp, self).__init__()
        self.dataset = dataset
        self.lambdas = lambdas
        self.Sub = Sub
        self.Mul = Mul
        self.add_loss = add_loss
        self.hidden_size = int(lambdas['roberta_hidden_size'])
        self.MLP_size = int(lambdas['MLP_size'])
        print("hidden_size: {}".format(self.hidden_size))
        print("MLP_size: {}".format(self.MLP_size))
        self.num_classes = num_classes
        self.model = RobertaModel.from_pretrained('roberta-base')
        if CUDA:
            self.hier_class_weights = torch.FloatTensor(hier_weights).cuda()
            self.temp_class_weights = torch.FloatTensor(temp_weights).cuda()
        else:
            self.hier_class_weights = torch.FloatTensor(hier_weights)
            self.temp_class_weights = torch.FloatTensor(temp_weights)
        self.HiEve_anno_loss = nn.CrossEntropyLoss(weight=self.hier_class_weights)
        self.MATRES_anno_loss = nn.CrossEntropyLoss(weight=self.temp_class_weights)
        self.transitivity_loss_H = transitivity_loss_H_()
        self.transitivity_loss_T = transitivity_loss_T_()
        self.cross_category_loss = cross_category_loss_()
        if freq is not None:
            self.fc1 = nn.Linear(self.hidden_size*4+1, self.MLP_size)  
        else:
            if self.Sub is None and self.Mul is None:
                self.fc1 = nn.Linear(self.hidden_size*2, self.MLP_size)
                self.fc2 = nn.Linear(self.MLP_size, num_classes)
            elif self.Sub is not None and self.Mul is not None:
                self.fc1 = nn.Linear(self.hidden_size*4, self.MLP_size*2)
                self.fc2 = nn.Linear(self.MLP_size*2, num_classes)
            else:
                self.fc1 = nn.Linear(self.hidden_size*3, int(self.MLP_size*1.5))
                self.fc2 = nn.Linear(int(self.MLP_size*1.5), num_classes)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x_sent, y_sent, z_sent, x_position, y_position, z_position, xy, yz, xz, flag, loss_out = None):
        batch_size = x_position.size(0)
        
        output_x = self.model(x_sent)[0]
        output_y = self.model(y_sent)[0]
        output_z = self.model(z_sent)[0]
        
        output_A = torch.cat([output_x[i, x_position[i].long(), :].unsqueeze(0) for i in range(0, batch_size)], 0)
        output_B = torch.cat([output_y[i, y_position[i].long(), :].unsqueeze(0) for i in range(0, batch_size)], 0)
        output_C = torch.cat([output_z[i, z_position[i].long(), :].unsqueeze(0) for i in range(0, batch_size)], 0)
        print(output_A.size())
        
        if self.Sub is None and self.Mul is None:
            alpha_representation = torch.cat((output_A, output_B), 1)
            beta_representation = torch.cat((output_B, output_C), 1)
            gamma_representation = torch.cat((output_A, output_C), 1)
        elif self.Sub is not None and self.Mul is not None:
            subAB = torch.sub(output_A, output_B)
            subBC = torch.sub(output_B, output_C)
            subAC = torch.sub(output_A, output_C)
            mulAB = torch.mul(output_A, output_B)
            mulBC = torch.mul(output_B, output_C)
            mulAC = torch.mul(output_A, output_C)
            alpha_representation = torch.cat((output_A, output_B, subAB, mulAB), 1)
            beta_representation = torch.cat((output_B, output_C, subBC, mulBC), 1)
            gamma_representation = torch.cat((output_A, output_C, subAC, mulAC), 1)
        elif self.Sub is not None and self.Mul is None:
            subAB = torch.sub(output_A, output_B)
            subBC = torch.sub(output_B, output_C)
            subAC = torch.sub(output_A, output_C)
            alpha_representation = torch.cat((output_A, output_B, subAB), 1)
            beta_representation = torch.cat((output_B, output_C, subBC), 1)
            gamma_representation = torch.cat((output_A, output_C, subAC), 1)
        else:
            mulAB = torch.mul(output_A, output_B)
            mulBC = torch.mul(output_B, output_C)
            mulAC = torch.mul(output_A, output_C)
            alpha_representation = torch.cat((output_A, output_B, mulAB), 1)
            beta_representation = torch.cat((output_B, output_C, mulBC), 1)
            gamma_representation = torch.cat((output_A, output_C, mulAC), 1)
        
        print(alpha_representation.size())
        alpha_logits = self.fc2(self.relu(self.fc1(alpha_representation)))
        beta_logits = self.fc2(self.relu(self.fc1(beta_representation)))
        gamma_logits = self.fc2(self.relu(self.fc1(gamma_representation)))
        
        if loss_out is None:
            return alpha_logits, beta_logits, gamma_logits
        else:
            loss = 0.0
            if self.dataset == "MATRES":
                loss += self.lambdas['lambda_annoT'] * self.MATRES_anno_loss(alpha_logits, xy) + self.MATRES_anno_loss(beta_logits, yz) + self.MATRES_anno_loss(gamma_logits, xz)
                if self.add_loss:
                    loss += self.lambdas['lambda_transT'] * self.transitivity_loss_T(alpha_logits, beta_logits, gamma_logits).sum()
            
            if self.dataset == "I2B2":
                loss += self.lambdas['lambda_annoT'] * self.MATRES_anno_loss(alpha_logits, xy) + self.MATRES_anno_loss(beta_logits, yz) + self.MATRES_anno_loss(gamma_logits, xz)
                if self.add_loss:
                    loss += self.lambdas['lambda_transT'] * self.transitivity_loss_T(alpha_logits, beta_logits, gamma_logits).sum()

            elif self.dataset == "HiEve":
                loss += self.lambdas['lambda_annoH'] * self.HiEve_anno_loss(alpha_logits, xy) + self.HiEve_anno_loss(beta_logits, yz) + self.HiEve_anno_loss(gamma_logits, xz)
                if self.add_loss:
                    loss += self.lambdas['lambda_transH'] * self.transitivity_loss_H(alpha_logits, beta_logits, gamma_logits).sum()

            elif self.dataset == "Joint":
                for i in range(0, batch_size):
                    if flag[i] == 1 or flag[i]==2:
                        loss += self.lambdas['lambda_annoT'] * (self.MATRES_anno_loss(alpha_logits[i][4:].view([1, 4]), xy[i].view(1)) + self.MATRES_anno_loss(beta_logits[i][4:].view([1, 4]), yz[i].view(1)) + self.MATRES_anno_loss(gamma_logits[i][4:].view([1, 4]), xz[i].view(1)))
                    elif flag[i] == 0:
                        loss += self.lambdas['lambda_annoH'] * (self.HiEve_anno_loss(alpha_logits[i][0:4].view([1, 4]), xy[i].view(1)) + self.HiEve_anno_loss(beta_logits[i][0:4].view([1, 4]), yz[i].view(1)) + self.HiEve_anno_loss(gamma_logits[i][0:4].view([1, 4]), xz[i].view(1)))
                if self.add_loss:
                    loss += self.lambdas['lambda_transT'] * self.transitivity_loss_T(alpha_logits[4:], beta_logits[4:], gamma_logits[4:]).sum()
                    loss += self.lambdas['lambda_transH'] * self.transitivity_loss_H(alpha_logits[0:4], beta_logits[0:4], gamma_logits[0:4]).sum()
                    if self.add_loss == 2:
                        loss += self.lambdas['lambda_cross'] * self.cross_category_loss(alpha_logits, beta_logits, gamma_logits).sum()
            else:
                raise ValueError("Currently not supporting this setting! -_-'")

            return alpha_logits, beta_logits, gamma_logits, loss
