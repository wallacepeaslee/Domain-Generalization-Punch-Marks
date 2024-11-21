# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

'''
The below is modified from the train.py script.
It saves the test and validation images, and their filepaths,
depending on the trial seed used (since each different trial seed
results in a different train/valid split).
'''


import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from torchvision.utils import save_image

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    # print('HParams:')
    # hparams['batch_size'] = 64
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    
    
    '''
    TODO: the below root_save should give a path to a folder when the given
    labled images and text files should be saved
    '''
    root_save = '/home/wep25/rds/hpc-work/Sprints/DG_Oct2024/Data/DataSplits/'
    print(os.getcwd)
    for env_i, env in enumerate(dataset):
        uda = []
        
        #print("env!", env)

        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))
        print("vv_Env!", env, "\n", len(out), len(in_))
        print(out.keys)
        envPathIn = os.path.join(root_save, 'env_' + str(env_i) + '_in')
        envPathOut = os.path.join(root_save, 'env_' + str(env_i) + '_out')
        print(envPathOut)
        print(os.getcwd())
        os.mkdir(envPathOut)
        os.mkdir(envPathIn)
        fpListOut = []
        fpListIn = []
        for elm in range(len(out.keys)):
            #print(elm)
            #print(elm, out.__getitem__(elm)[0].shape, type(out.__getitem__(elm)[1]), end = ';')
            fpSave = os.path.join(envPathOut, 'img_out_Number_' + str(elm) + '_key_' + str(out.keys[elm]) + '_class_' + str(out.__getitem__(elm)[1]) + '_img.tif')
            fpListOut.append((fpSave, out.__getitem__(elm)[1]))
            print(fpSave)
            save_image(out.__getitem__(elm)[0], fpSave)
            #break
        for elm in range(len(in_.keys)):
            #print(elm)
            #print(elm, in_.__getitem__(elm)[0].shape, type(in_.__getitem__(elm)[1]), end = ';')
            fpSave = os.path.join(envPathIn, 'img_in_Number_' + str(elm) + '_key_' + str(in_.keys[elm]) + '_class_' + str(in_.__getitem__(elm)[1]) + '_img.tif')
            fpListIn.append((fpSave, in_.__getitem__(elm)[1]))
            print(fpSave)
            save_image(in_.__getitem__(elm)[0], fpSave)
            #break
        
        strOut = ''
        for elm in fpListOut:
            strOut += elm[0] + ' ' + str(elm[1]) + '\n'
        strIn = ''
        for elm in fpListIn:
            strIn += elm[0] + ' ' + str(elm[1]) + '\n'
        
        txtOut = os.path.join(root_save, 'fp_out_' + str(env_i) + '.txt')
        txtIn = os.path.join(root_save, 'fp_in_' + str(env_i) + '.txt')
        with open(txtOut, 'a') as f:
            f.write(strOut)
        with open(txtIn, 'a') as f:
            f.write(strIn)

