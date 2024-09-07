import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
import os 

# storage
parser.add_argument("--dataset", type=str, default="engagement")
parser.add_argument("--data_root", type=str, default="../data")
parser.add_argument("--label_root", type=str, default="label_0402_3k")
parser.add_argument("--ckpt_root", type=str, default="checkpoints")
parser.add_argument("--ckpt_name", type=str, required=True)

# data
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--num_labels", type=int, default=3)

# video
parser.add_argument("--num_frames", type=int, default=16)

# audio
parser.add_argument("--sampling_rate", type=int, default=16000)
parser.add_argument("--max_audio_len", type=int, default=160000)

# training
parser.add_argument("--model", type=str, default="early-fusion")
parser.add_argument("--num_epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--hidden_size", type=float, default=128)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--clip", type=float, default=0.1)
parser.add_argument("--patience", type=int, default=10, help="early stopping")
parser.add_argument("--device", type=int, default=-2, help="gpu device")
parser.add_argument("--targettype", type=str, default="ce", help="mse or ce")
parser.add_argument("--model_freeze", type=str, default="part", help="model freeze or not")
parser.add_argument("--activation_fn", type=str, default="tanh", help="activation function")
parser.add_argument("--name", type=str, default="default", help="wandb name")
#extra_dropout
parser.add_argument("--extra_dropout", type=int, default=0)
parser.add_argument("--kfolds", type=int, default=0)
parser.add_argument("--trainmets", type=int, default=0)
#scheduler
parser.add_argument("--scheduler", type=str, default='none', help="scheduler, none or linear")
parser.add_argument("--openfacefeat", type=int, default=0)
parser.add_argument("--openfacefeat_extramlp", type=int, default=0)
parser.add_argument("--openfacefeat_extramlp_dim", type=int, default=0)

parser.add_argument("--ablation", type=int, default=0)

parser.add_argument("--expnum", type=int, default=0)
config = parser.parse_args()
print(config)

if config.device != -2:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(config.device)
import torch
import random
import numpy as np

from solver_base import solver_base

def set_seed(seed):
    # Reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    random.seed(seed)
    np.random.seed(seed)

# Fix random seed
set_seed(config.seed)

# Setup solver
solver = solver_base(config).cuda()
import wandb
wandb.init( name = config.name, config=config)

# Start training
solver.run_eval()
