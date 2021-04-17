import tqdm
import time
import random
import torch 
import sys
from models.joint_constrain_model import *
from data_loader.data_loaders import joint_constrained_loader
from Exp import EXP


torch.manual_seed(42)
cuda = torch.device('cuda')
# Read parameters
rst_file_name = "00001.rst"
# Restore model
model_params_dir = "./model_params/"
HiEve_best_PATH = model_params_dir + "HiEve_best/" + rst_file_name.replace(".rst", ".pt")
MATRES_best_PATH = model_params_dir + "MATRES_best/" + rst_file_name.replace(".rst", ".pt")
I2B2_best_PATH = model_params_dir + "I2B2_best/" + rst_file_name.replace(".rst", ".pt") 

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

dataset = "Joint"
add_loss = 1
epochs = 32
batch_size = 16

# Use Optuna to select the best hyperparameters
import optuna
from timeit import default_timer as timer
interaction = 0
def objective(trial):    
    params ={
        "downsample": trial.suggest_float("downsample", 0.01, 0.2),
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True),
        'lambda_annoT': trial.suggest_float('lambda_annoT', 0.0, 1.0),
        'lambda_annoH': trial.suggest_float('lambda_annoH', 0.0, 1.0),
        'lambda_transT': trial.suggest_float('lambda_transT', 0.0, 1.0),
        'lambda_transH': trial.suggest_float('lambda_transH', 0.0, 1.0),
        'lambda_cross': trial.suggest_float('lambda_cross', 0.0, 1.0),
        'MLP_size': trial.suggest_categorical("MLP_size", [512, 256, 768]),
        'num_layers': trial.suggest_int("num_layers", 1, 3),
        'lstm_hidden_size': trial.suggest_categorical("lstm_hidden_size", [512, 256]),
        'roberta_hidden_size': trial.suggest_categorical("roberta_hidden_size", [768, 1024]),
        'lstm_input_size': 768,
    }

    global interaction
    interaction += 1
    start = timer()
    train_dataloader, valid_dataloader_MATRES, test_dataloader_MATRES, valid_dataloader_HIEVE, test_dataloader_HIEVE, valid_dataloader_I2B2, test_dataloader_I2B2, num_classes = joint_constrained_loader(dataset, params['downsample'], batch_size)
    
    model = roberta_mlp(num_classes, dataset, add_loss, space)
    model.to(cuda)
    model.zero_grad()
    print("# of parameters:", count_parameters(model))
    model_name = rst_file_name.replace(".rst", "") # to be designated after finding the best parameters
    total_steps = len(train_dataloader) * epochs
    print("Total steps: [number of batches] x [number of epochs] =", total_steps)

    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    if dataset == "MATRES":
        total_steps = len(train_dataloader) * epochs
        print("Total steps: [number of batches] x [number of epochs] =", total_steps)
        matres_exp = EXP(cuda, model, epochs, params['learning_rate'], train_dataloader, valid_dataloader_MATRES, test_dataloader_MATRES, None, None, None, None, finetune, dataset, MATRES_best_PATH, None, None, None, model_name)
        T_F1, H_F1, I_F1 = matres_exp.train()
        matres_exp.evaluate(eval_data = "MATRES", test = True)
    if dataset == "I2B2":
        total_steps = len(train_dataloader) * epochs
        print("Total steps: [number of batches] x [number of epochs] =", total_steps)
        i2b2_exp = EXP(cuda, model, epochs, params['learning_rate'], train_dataloader, valid_dataloader_I2B2, test_dataloader_I2B2, None, None, finetune, dataset, None, I2B2_best_PATH, None, None, model_name)
        T_F1, H_F1, I_F1 = i2b2_exp.train()
        i2b2_exp.evaluate(eval_data = "I2B2", test = True)
    elif dataset == "HiEve":
        total_steps = len(train_dataloader) * epochs
        print("Total steps: [number of batches] x [number of epochs] =", total_steps)
        hieve_exp = EXP(cuda, model, epochs, params['learning_rate'], train_dataloader, None, None, valid_dataloader_HIEVE, test_dataloader_HIEVE, finetune, dataset, None, None, HiEve_best_PATH, None, model_name)
        T_F1, H_F1, I_F1 = hieve_exp.train()
        hieve_exp.evaluate(eval_data = "HiEve", test = True)
    elif dataset == "Joint":
        total_steps = len(train_dataloader) * epochs
        print("Total steps: [number of batches] x [number of epochs] =", total_steps)
        joint_exp = EXP(cuda, model, epochs, params['learning_rate'], train_dataloader, valid_dataloader_MATRES, test_dataloader_MATRES, valid_dataloader_HIEVE, test_dataloader_HIEVE, finetune, dataset, MATRES_best_PATH, HiEve_best_PATH, None, model_name)
        T_F1, H_F1, I_F1 = joint_exp.train()
        joint_exp.evaluate(eval_data = "HiEve", test = True)
        joint_exp.evaluate(eval_data = "MATRES", test = True)
        joint_exp.evaluate(eval_data="I2B2", test=True)
    else:
        raise ValueError("Currently not supporting this dataset! -_-'")
    
    print(f'Iteration {ITERATION} result: MATRES F1: {T_F1}; HiEve F1: {H_F1}; I2B2 F1: {I_F1}')
    
    run_time = format_time(timer() - start)
    
    # Write to the csv file ('a' means append)
    print("########################## Append a row to out_file ##########################")
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, T_F1, H_F1, params, ITERATION, run_time])

    return T_F1, H_F1, I_F1

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)
trial = study.best_trial

print('Accuracy: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))