import argparse
import numpy as np


def init_args(): 
    parser = argparse.ArgumentParser(description="""Configure""")

    # basic configuration 
    parser.add_argument('--exp', type=str, default='exp0',
                        help='experiment folder')

    parser.add_argument('--epochs', type=int, default=30,
                        help='number of total epochs to run (default: 90)')

    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--resume', default='', type=str,
                        metavar='PATH', help='path to checkpoint (default: None)')
    parser.add_argument('--save_step', default=5, type=int)
    parser.add_argument('--valid_step', default=1, type=int)

    # Dataloader config
    parser.add_argument('--max_sample', default=-1, type=int)
    parser.add_argument('--repeat', default=1, type=int)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--audio_len', type=float, default=1, choices=[1, 3])

    # model config
    parser.add_argument('--setting', type=str, default='music_multi_nodes', required=False)
    parser.add_argument('--mix_audio_nodes', type=int, default=2, required=False)
    parser.add_argument('--cycle_temp', type=float, default=0.07, required=False)
    
    
    # optimizer parameters
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--lr_scheduler', default=0, type=float, help='whether to have learning rate scheduler')
    parser.add_argument('--lr_decay', type=int, default=3, metavar='LRDECAY',
            help=('Multiply the learning rate by lr-decay-multiplier every lr-decay'
                  ' number of epochs'))
    parser.add_argument('--lr_decay_multiplier', type=float, default=0.95,
            metavar='LRDECAYMULT',
            help='Multiply the learning rate by this factor every lr-decay epochs')

    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', default=5e-4,
                        type=float, help='weight decay (default: 5e-4)')
    parser.add_argument('--optim', type=str, default='Adam',
                        choices=['SGD', 'Adam'])

    args = parser.parse_args()
    return args
