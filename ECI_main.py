from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from models.roberta_model import ECIRoberta
from numpy import sin
import torch
import optuna
from timeit import default_timer as timer
from data_loader.data_loaders import single_loader
from ECIExp import EXP
from utils.tools import format_time
from utils.constant import CUDA
from torchsummary import summary

def objective(trial:optuna.Trial):
    params = {
        "learning_rate": trial.suggest_float("lr", 1e-6, 1e-2, log=True),
        "MLP size": trial.suggest_categorical("MLP size", [256, 512, 768])
    }
    start = timer()
    train_dataloader, test_dataloader, validate_dataloader, num_classes = single_loader(dataset, batch_size)

    model = ECIRoberta(num_classes, dataset, params["MLP size"], roberta_type)
    if CUDA:
        model = model.cuda()
    model.zero_grad()
    total_steps = len(train_dataloader) * epoches
    print("Total steps: [number of batches] x [number of epochs] =", total_steps)

    exp = EXP(model, epoches, params["learning_rate"], train_dataloader, validate_dataloader, test_dataloader, best_path)
    f1 = exp.train()
    exp.evaluate(is_test=True)
    
    print("Result: Best micro F1 of interaction: {}".format(f1))
    return f1

if __name__=="__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='SEED', default=0, type=int)
    parser.add_argument('--batch_size', help="Batch size", default=32, type=int)
    parser.add_argument('--roberta_type', help="base or large", default='base', type=str)
    parser.add_argument('--epoches', help='Number epoch', default=30, type=int)
    parser.add_argument('--best_path', help="Path for save model", type=str)
    parser.add_argument('--dataset', help="Path for dataset", type=str)

    args = parser.parse_args()
    seed = args.seed
    batch_size = args.batch_size
    roberta_type  = args.roberta_type
    epoches = args.epoches
    best_path = args.best_path
    dataset = args.dataset

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    trial = study.best_trial

    print('Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))

