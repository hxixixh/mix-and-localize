from collections import OrderedDict
import os
import pprint
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import init_args, params
from data.music_img_aud_pair import MusicImg2AudPairDataset
from data.voxceleb_img_aud_pair import VoxCelebImg2AudPairDataset
from models import MixAudModelFeatMultiAud, MixAudSIS1FeatHardCycleLoss

DEVICE = torch.device("cuda")


def load_model(cp_path, net, strict=True): 
    if os.path.isfile(cp_path): 
        print("=> loading checkpoint '{}'".format(cp_path))
        checkpoint = torch.load(cp_path)

        if list(checkpoint['state_dict'].keys())[0][:7] == 'module.': 
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items(): 
                name = k[7:]
                new_state_dict[name] = v
            net.load_state_dict(new_state_dict, strict=strict) 
        else: 
            net.load_state_dict(checkpoint['state_dict'], strict=strict)
        
        print("=> loaded checkpoint '{}' (epoch {})"
                    .format(cp_path, checkpoint['epoch']))
        start_epoch = checkpoint['epoch']
    else: 
        print("=> no checkpoint found at '{}'".format(cp_path))
        start_epoch = 0
    return net, start_epoch


def load_model_and_opt(cp_path, net, optimizer, strict=True, load_opt=True): 
    net, start_epoch = load_model(cp_path, net, strict=strict)
    print("=> loading optimizer")
    if load_opt: 
        optim_state = torch.load(cp_path)['optimizer']
        optimizer.load_state_dict(optim_state)
        print("=> loaded optimizer")

    return net, start_epoch, optimizer


def adjust_learning_rate(base_lr, lr_decay, lr_decay_multiplier, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed every lr_decay epochs"""
    lr = base_lr * (lr_decay_multiplier ** (epoch // lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    lr_check = optimizer.param_groups[0]['lr']
    return lr, lr_check


def get_dataloader(args, pr):
    if args.setting == 'music_multi_nodes': 
        train_dataset = MusicImg2AudPairDataset(args, pr, pr.list_train, split='train')
        val_dataset = MusicImg2AudPairDataset(args, pr, pr.list_val, split='val')

    elif args.setting == 'voxceleb_multi_nodes': 
        train_dataset = VoxCelebImg2AudPairDataset(args, pr, pr.list_train, split='train')
        val_dataset = VoxCelebImg2AudPairDataset(args, pr, pr.list_val, split='val')

    drop_last_val = False
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=True)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=drop_last_val)
    
    return train_loader, val_loader


def make_optimizer(model, args):
    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=False
        )
    elif args.optim == 'Adam':
        args.lr = args.lr / 10
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    return optimizer


def validation(args, pr, net, criterion, data_loader, device='cuda', epoch=0):
    net.eval()
    with torch.no_grad():
        loss_dict = {}
        acc_dict = {}
        for step, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Validation"): 
            img, audio, audio_2 = batch['frames'].to(device), batch['audio'].to(device), batch['audio_2'].to(device)
            audio_1_mix = audio.reshape(audio.shape[0] // args.mix_audio_nodes, args.mix_audio_nodes, -1).mean(1)
            audio_2_mix = audio_2.reshape(audio_2.shape[0] // args.mix_audio_nodes, args.mix_audio_nodes, -1).mean(1)
            audio = torch.cat([audio_1_mix, audio_2_mix], dim=0)

            out = net(img, audio)
            loss, diagnostics = criterion.compute_loss(out, max_mode=pr.max_mode)

            for i, key in enumerate(diagnostics.keys()): 
                if 'xent' in key: 
                    if step == 0: 
                        loss_dict[key] = 0
                    loss_dict[key] += diagnostics[key].mean().item()
                elif 'acc' in key: 
                    if step == 0: 
                        acc_dict[key] = 0
                    acc_dict[key] += diagnostics[key].mean().item()
        
        for i, key in enumerate(loss_dict.keys()): 
            loss_dict[key] = loss_dict[key] / len(data_loader)

        for i, key in enumerate(acc_dict.keys()):  
            acc_dict[key] = acc_dict[key] / len(data_loader)

    return loss_dict, acc_dict


def main(args, device):
    gpus = torch.cuda.device_count()
    gpu_ids = list(range(gpus))

    # ----- make dirs for checkpoints ----- #
    if not os.path.exists('./checkpoints/' + args.exp):
        os.makedirs('./checkpoints/' + args.exp)
    # tensorboard
    writer = SummaryWriter(os.path.join(
        './checkpoints', args.exp, 'visualization'))
    # ------------------------------------- #

    # ----- get parameters for audio ------ #
    fn = getattr(params, args.setting)
    pr = fn()
    args_nice = pprint.pformat(vars(args))
    pr_nice = pprint.pformat(vars(pr))
    with open(os.path.join('./checkpoints', args.exp, 'parameters.txt'), "a") as f: 
        f.write(args_nice)
        f.write("\n")
        f.write(pr_nice)
    # ------------------------------------- #

    # ----- Dataset and Dataloader ----- #
    train_loader, val_loader = get_dataloader(args, pr)
    net = MixAudModelFeatMultiAud(pr, num_node=args.mix_audio_nodes).to(device)
    
    pr.mean_max_mode = False 
    pr.max_mode = True
    criterion = MixAudSIS1FeatHardCycleLoss(args.mix_audio_nodes, cycle_temp=args.cycle_temp)

    # ----- Optimizer ----- #
    optimizer = make_optimizer(net, args)

    # -------- Loading checkpoints weights ------------- #
    if args.resume: 
        net, args.start_epoch, optimizer = load_model_and_opt(args.resume, net, optimizer, strict=False, load_opt=False)

    if len(gpu_ids) > 1: 
        net = nn.DataParallel(net, device_ids=gpu_ids)
    
    loss_dict, acc_dict = validation(args, pr, net, criterion, val_loader, device, epoch=args.start_epoch)
    writer.add_scalars('/validation loss', loss_dict, 0)
    writer.add_scalars('/validation acc', acc_dict, 0)
    tqdm.write("Initial, Validation Loss: {}".format(loss_dict))
    tqdm.write('Acc: {}'.format(acc_dict))
    tqdm.write('\n')

    for epoch in range(args.start_epoch, args.epochs):
        if args.lr_scheduler: 
            cur_lr, lr_check = adjust_learning_rate(args.lr, args.lr_decay, 
                                        args.lr_decay_multiplier,
                                        optimizer, epoch)
            print('Learning rate @ %5d is %f (expected %f)' % (epoch, lr_check, cur_lr))

        net.train()

        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
            img, audio, audio_2 = batch['frames'].to(device), batch['audio'].to(device), batch['audio_2'].to(device)
            audio_1_mix = audio.reshape(audio.shape[0] // args.mix_audio_nodes, args.mix_audio_nodes, -1).mean(1)
            audio_2_mix = audio_2.reshape(audio_2.shape[0] // args.mix_audio_nodes, args.mix_audio_nodes, -1).mean(1)
            audio = torch.cat([audio_1_mix, audio_2_mix], dim=0)
        
            out = net(img, audio)

            loss, diagnostics = criterion.compute_loss(out, max_mode=pr.max_mode)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_step = epoch * len(train_loader) + step + 1

            BOARD_STEP = 20
            if (step+1) % BOARD_STEP == 0:
                loss_tracker = {}
                acc_tracker = {}
                for i, key in enumerate(diagnostics.keys()):
                    if 'xent' in key: 
                        loss_tracker[key] = diagnostics[key].mean().item()
                    elif 'acc' in key: 
                        acc_tracker[key] = diagnostics[key].mean().item()
                    writer.add_scalars('/training loss', loss_tracker, current_step)
                    writer.add_scalars('/training acc', acc_tracker, current_step)
                    tqdm.write("Epoch: {}/{}, step: {}/{}, loss: {}, acc: {}".format(epoch+1, args.epochs, step+1, len(train_loader), loss_tracker, acc_tracker))

        # ----------- Validtion -------------- #
        VALID_STEP = args.valid_step
        if (epoch + 1) % VALID_STEP == 0:
            loss_dict, acc_dict = validation(args, pr, net, criterion, val_loader, device)
            writer.add_scalars('/validation loss', loss_dict, epoch + 1)
            writer.add_scalars('/validation acc', acc_dict, epoch + 1)
            tqdm.write("Epoch: {}/{}, Validation Loss: {}".format(epoch + 1, args.epochs, loss_dict))
            tqdm.write('Acc: {}'.format(acc_dict))
            tqdm.write('\n')
        
        # ---------- Save model ----------- #
        SAVE_STEP = args.save_step
        if (epoch + 1) % SAVE_STEP == 0:
            path = os.path.join('./checkpoints', args.exp, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar')
            torch.save({'epoch': epoch + 1,
                        'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        },
                        path)
         # --------------------------------- #
    tqdm.write('Training Complete!')
    writer.close()


if __name__ == '__main__':
    args = init_args()
    main(args, DEVICE)

