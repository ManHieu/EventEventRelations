from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import datetime
from models.roberta_model import ECIRoberta
from numpy import sin
import torch
import optuna
from timeit import default_timer as timer
from data_loader.data_loaders import single_loader
from ECIExp import EXP
from utils.tools import format_time
from utils.constant import CUDA
# from torchsummary import summary

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def objective(trial:optuna.Trial):
    params = {
        "bert_learning_rate": trial.suggest_float("b_lr", 1e-7, 1e-6, step=1e-7),
        "mlp_learning_rate":trial.suggest_float("m_lr", 1e-5, 1e-4, step=1e-5),
        "MLP size": trial.suggest_categorical("MLP size", [512, 768]),
        "epoches": trial.suggest_categorical("epoches", [3, 6, 5]),
        'weight_decay': trial.suggest_float("weight_decay", 0, 0.4, step=0.2),
        'negative_slope': trial.suggest_float("negative_slope", 0, 0.5, step=0.1),
        'warmup_proportion': 0.1,
    }
    print("Hyperparameter will be used in this trial: ")
    print(params)
    start = timer()
    train_dataloader, test_dataloader, validate_dataloader, num_classes = single_loader(dataset, batch_size)

    model = ECIRoberta(num_classes, dataset, params["MLP size"], 
                    roberta_type, finetune=True, negative_slope=params["negative_slope"])
    if CUDA:
        model = model.cuda()
    model.zero_grad()
    # epoches = params['epoches']
    print("# of parameters:", count_parameters(model))
    total_steps = len(train_dataloader) * epoches
    print("Total steps: [number of batches] x [number of epochs] =", total_steps)

    exp = EXP(model, epoches, params["bert_learning_rate"], params["mlp_learning_rate"], 
            train_dataloader, validate_dataloader, test_dataloader, 
            best_path, weight_decay=params['weight_decay'], 
            train_lm_epoch=params['epoches'], warmup_proportion=params['warmup_proportion'])
    f1, CM = exp.train()
    exp.evaluate(is_test=True)
    
    print("Result: Best micro F1 of interaction: {}".format(f1))

    with open("{}{}.txt".format(result_folder, dataset), 'a', encoding='UTF-8') as f:
        f.write("\n -------------------------------------------- \n")
        f.write(" F1: \n {} \n CM: \n{} \n Hypeparameter: \n {} \n ".format(f1, CM, params))
        f.write("Time: {} \n".format(datetime.datetime.now()))
    return f1

if __name__=="__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='SEED', default=0, type=int)
    parser.add_argument('--batch_size', help="Batch size", default=32, type=int)
    parser.add_argument('--roberta_type', help="base or large", default='roberta-base', type=str)
    parser.add_argument('--epoches', help='Number epoch', default=30, type=int)
    parser.add_argument('--best_path', help="Path for save model", type=str)
    parser.add_argument('--dataset', help="Name of dataset", type=str)
    parser.add_argument('--result_log', help="Path of result folder",default='./result/', type=str)

    args = parser.parse_args()
    seed = args.seed
    batch_size = args.batch_size
    roberta_type  = args.roberta_type
    epoches = args.epoches
    best_path = args.best_path
    dataset = args.dataset
    result_folder = args.result_log

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    trial = study.best_trial

    print('Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))

