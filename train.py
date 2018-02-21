from __future__ import print_function
import sys
import os
import argparse
import numpy as np
sys.path.remove('/data/software/opencv-3.4.0/lib/python2.7/dist-packages')
import cv2
from datetime import datetime

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable

from lib.modeling.model_builder import create_model
from lib.dataset.dataset_factory import load_data
from lib.utils.config_parse import cfg_from_file
from lib.ssds_train import train_model

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a ssds.pytorch network')
    parser.add_argument('--cfg', dest='config_file',
            help='optional config file', default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

# def main():
#     # parse args from the  confg_file
#     args = parse_args()
#     if args.confg_file is not None:
#         cfg_from_file(args.config_file)

#     # Load data
#     train_loader = load_data(cfg.DATASET, 'train')
#     test_loader = load_data(cfg.DATASET, 'test')

#     model, priors = create_model(cfg.MODEL)

#     if 

    

def train():
    args = parse_args()
    if args.config_file is not None:
        cfg_from_file(args.config_file)
    train_model()

if __name__ == '__main__':
    train()
