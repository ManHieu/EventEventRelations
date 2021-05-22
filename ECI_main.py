from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from models.roberta_model_multi import ECIRobertaJointTask
from data_loader.EventDataset import EventDataset
import datetime
from torch.utils.data.dataloader import DataLoader
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
        "bert_learning_rate": trial.suggest_categorical("b_lr", [5e-8, 1e-7, 5e-7, 7e-7, 3e-7]),
        "mlp_learning_rate":trial.suggest_categorical("m_lr", [5e-6, 1e-5, 5e-5]),
        "MLP size": trial.suggest_categorical("MLP size", [768]),
        "epoches": trial.suggest_categorical("epoches", [1, 3, 5]),
        "b_lambda_scheduler": trial.suggest_categorical("b_scheduler", ['cosin']),
        "m_step": trial.suggest_int('m_step', 1, 3),
        'b_lr_decay_rate': trial.suggest_float('decay_rate', 0.5, 0.8, step=0.1)
    }
    print("Hyperparameter will be used in this trial: ")
    print(params)
    start = timer()
    train_set = []
    test_set = []
    val_set = []
    for dataset in datasets:
        train, test, validate = single_loader(dataset)
        train_set.extend(train)
        val_set.extend(validate)
        test_set.extend(test)
    train_dataloader = DataLoader(EventDataset(train_set), batch_size=batch_size, shuffle=True)
    validate_dataloader = DataLoader(EventDataset(val_set), batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(EventDataset(test_set), batch_size=batch_size, shuffle=True)

    model = ECIRobertaJointTask(params['MLP size'], roberta_type, datasets, finetune=True)
    # ECIRoberta(num_classes, dataset, params['MLP size'], roberta_type, finetune=True)
    if CUDA:
        model = model.cuda()
    model.zero_grad()
    # epoches = params['epoches']
    print("# of parameters:", count_parameters(model))
    total_steps = len(train_dataloader) * epoches
    print("Total steps: [number of batches] x [number of epochs] =", total_steps)

    exp = EXP(model, epochs=epoches, b_lr=params['bert_learning_rate'], m_lr=params['mlp_learning_rate'],
            decay_rate=params['b_lr_decay_rate'], m_lr_step=params['m_step'], b_scheduler_lambda=params['b_lambda_scheduler'],
            train_dataloader=train_dataloader, validate_dataloader=validate_dataloader, test_dataloader=test_dataloader,
            best_path=best_path, train_lm_epoch=params['epoches'])
    f1, CM = exp.train()
    exp.evaluate(is_test=True)
    
    print("Result: Best micro F1 of interaction: {}".format(f1))

    with open(result_file, 'a', encoding='UTF-8') as f:
        f.write("\n -------------------------------------------- \n")
        f.write("\n cosin_lr_lambda - decay rate 1.5 \n")
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
    parser.add_argument('--dataset', help="Name of dataset", type=list)
    parser.add_argument('--result_log', help="Path of result folder", type=str)

    args = parser.parse_args()
    seed = args.seed
    batch_size = args.batch_size
    roberta_type  = args.roberta_type
    epoches = args.epoches
    best_path = args.best_path
    datasets = args.dataset
    result_file = args.result_log

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    trial = study.best_trial

    print('Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))

