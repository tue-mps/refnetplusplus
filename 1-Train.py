import json
import argparse
import torch
import random
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import pkbar
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import time
from datetime import datetime
from model.refnetplusplus import refnetplusplus
from dataset.encoder import ra_encoder
from dataset.dataset_fusion import RADIal
from dataset.dataloader_fusion import CreateDataLoaders
from loss import pixor_loss
from utils.evaluation import run_evaluation
from utils.metrics_bev import count_params

def main(config):

    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    # create experience name
    curr_date = datetime.now()
    exp_name = config['name'] + '___' + curr_date.strftime('%b-%d-%Y___%H:%M:%S')
    print(exp_name)
    st = time.time()

    # Create directory structure
    output_folder = Path(config['output']['dir'])
    output_folder.mkdir(parents=True, exist_ok=True)
    (output_folder / exp_name).mkdir(parents=True, exist_ok=True)
    # and copy the config file
    with open(output_folder / exp_name / 'config_fusion.json', 'w') as outfile:
        json.dump(config, outfile)

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device Used:', device)
    # Initialize tensorboard
    writer = SummaryWriter(output_folder / exp_name)

    # load dataset and create model
    enc = ra_encoder(geometry=config['dataset']['geometry'],
                     statistics=config['dataset']['statistics'],
                     regression_layer=2)

    if config['architecture']['bev']['refnetplusplus'] == 'True':
        dataset = RADIal(config=config,
                         encoder=enc.encode,
                         difficult=True)

        train_loader, val_loader, test_loader = CreateDataLoaders(dataset, config, config['seed'])

        net = refnetplusplus(mimo_layer=config['model']['MIMO_output'],
                          channels=config['model']['channels'],
                          channels_bev=config['model']['channels_bev'],
                          blocks=config['model']['backbone_block'],
                          detection_head=config['model']['DetectionHead'],
                          segmentation_head=config['model']['SegmentationHead'],
                          config=config, regression_layer=2)

        print("**************************")
        print("REFNet++ started to train!")
        print("**************************")

    print('Number of trainable parameters in the model: %s' % str(count_params(net) / 1e6))
    net.to(device)

    # Optimizer
    lr = float(config['optimizer']['lr'])
    step_size = int(config['lr_scheduler']['step_size'])
    gamma = float(config['lr_scheduler']['gamma'])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    num_epochs=int(config['num_epochs'])

    print('===========  Optimizer  ==================:')
    print('      LR:', lr)
    print('      step_size:', step_size)
    print('      gamma:', gamma)
    print('      num_epochs:', num_epochs)
    print('')

    # Train
    startEpoch = 0
    global_step = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'mAP': [], 'mAR': [], 'mIoU': []}
    freespace_loss = nn.BCEWithLogitsLoss(reduction='mean')

    # Set up early stopping parameters
    patience = 8  # Number of epochs with no improvement after which training will be stopped
    early_stopping_counter = 0
    best_validation_loss = float('inf')
    classif_loss = torch.tensor(0, dtype=torch.float64)
    reg_loss = torch.tensor(0, dtype=torch.float64)
    loss_seg = torch.tensor(0, dtype=torch.float64)

    for epoch in range(startEpoch,num_epochs):
        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)

        ###################
        ## Training loop ##
        ###################
        net.train()
        running_loss = 0.0

        for i, data in enumerate(train_loader):
            is_training = True
            inputs1 = data[0].to(device).float() #radar data
            inputs2 = data[1].to(device).float() #camera half fv image
            seg_map_label = data[2].to(device).double()
            det_label = data[3].to(device).float()
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = net(inputs2, inputs1, is_training)

            if config['model']['DetectionHead']=='True':
                classif_loss, reg_loss = pixor_loss(outputs['Detection'], det_label, config['losses'],config['model'])
                classif_loss *= config['losses']['weight'][0]
                reg_loss *= config['losses']['weight'][1]
            if config['model']['SegmentationHead'] == 'True':
                prediction = outputs['Segmentation'].contiguous().flatten()
                label = seg_map_label.contiguous().flatten()
                loss_seg = freespace_loss(prediction, label)
                loss_seg *= inputs1.size(0)
                loss_seg *= config['losses']['weight'][2]

            loss = classif_loss + reg_loss + loss_seg
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Loss/train_clc', classif_loss.item(), global_step)
            writer.add_scalar('Loss/train_freespace', loss_seg.item(), global_step)
            writer.add_scalar('Loss/train_reg', reg_loss.item(), global_step)
            # backprop
            loss.backward()
            optimizer.step()
            # statistics
            running_loss += loss.item() * inputs1.size(0)
            if config['model']['DetectionHead'] == 'True' and config['model']['SegmentationHead'] == 'True':
                kbar.update(i, values=[("loss", loss.item()), ("class", classif_loss.item()), ("reg", reg_loss.item()),
                                       ("freeSpace", loss_seg.item())])
            if config['model']['DetectionHead'] == 'False' and config['model']['SegmentationHead'] == 'True':
                kbar.update(i, values=[("freeSpace", loss_seg.item())])
            if config['model']['DetectionHead'] == 'True' and config['model']['SegmentationHead'] == 'False':
                kbar.update(i, values=[("loss", loss.item()), ("class", classif_loss.item()), ("reg", reg_loss.item())])

            global_step += 1

        scheduler.step()

        history['train_loss'].append(running_loss / len(train_loader.dataset))
        history['lr'].append(scheduler.get_last_lr()[0])

        ######################
        ## validation phase ##
        ######################

        eval = run_evaluation(net=net,loader=val_loader,
                              device=device,config=config,encoder=enc,
                              detection_loss=pixor_loss,
                              segmentation_loss=freespace_loss,
                              losses_params=config['losses'],
                              mode_params=config['model'])


        if config['model']['DetectionHead'] == 'True' and config['model']['SegmentationHead'] == 'True':
            history['val_loss'].append(eval['loss'])
            history['mAP'].append(eval['mAP'])
            history['mAR'].append(eval['mAR'])
            history['mIoU'].append(eval['mIoU'])

            kbar.add(1, values=[("val_loss", eval['loss']),("mAP", eval['mAP']),("mAR", eval['mAR']),("mIoU", eval['mIoU'])])

            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar('Loss/test', eval['loss'], global_step)
            writer.add_scalar('Metrics/mAP', eval['mAP'], global_step)
            writer.add_scalar('Metrics/mAR', eval['mAR'], global_step)
            writer.add_scalar('Metrics/mIoU', eval['mIoU'], global_step)

            # Saving all checkpoint as the best checkpoint for multi-task is a balance between both --> up to the user to decide
            name_output_file = config['name']+'_epoch{:02d}_loss_{:.4f}_AP_{:.4f}_AR_{:.4f}_IOU_{:.4f}.pth'.format(epoch, eval['loss'],eval['mAP'],eval['mAR'],eval['mIoU'])

        if config['model']['DetectionHead'] == 'False' and config['model']['SegmentationHead'] == 'True':
            history['val_loss'].append(eval['loss'])
            history['mIoU'].append(eval['mIoU'])

            kbar.add(1, values=[("val_loss", eval['loss']),
                                ("mIoU", eval['mIoU'])])

            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar('Loss/test', eval['loss'], global_step)
            writer.add_scalar('Metrics/mIoU', eval['mIoU'], global_step)

            # Saving all checkpoint as the best checkpoint for multi-task is a balance between both --> up to the user to decide
            name_output_file = config['name'] + '_epoch{:02d}_loss_{:.4f}_IOU_{:.4f}.pth'.format(
                epoch, eval['loss'], eval['mIoU'])

        if config['model']['DetectionHead'] == 'True' and config['model']['SegmentationHead'] == 'False':
            history['val_loss'].append(eval['loss'])
            history['mAP'].append(eval['mAP'])
            history['mAR'].append(eval['mAR'])

            kbar.add(1, values=[("val_loss", eval['loss']),("mAP", eval['mAP']),("mAR", eval['mAR'])])

            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar('Loss/test', eval['loss'], global_step)
            writer.add_scalar('Metrics/mAP', eval['mAP'], global_step)
            writer.add_scalar('Metrics/mAR', eval['mAR'], global_step)

            # Saving all checkpoint as the best checkpoint for multi-task is a balance between both --> up to the user to decide
            name_output_file = config['name']+'_epoch{:02d}_loss_{:.4f}_AP_{:.4f}_AR_{:.4f}.pth'.format(epoch, eval['loss'],eval['mAP'],eval['mAR'])

        filename = output_folder / exp_name / name_output_file

        checkpoint={}
        checkpoint['net_state_dict'] = net.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['scheduler'] = scheduler.state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['history'] = history
        checkpoint['global_step'] = global_step

        torch.save(checkpoint,filename)

        et = time.time()
        elapsed_time_seconds = et - st
        elapsed_time_minutes = elapsed_time_seconds / 60
        print('Total time consumed so far in minutes:', elapsed_time_minutes, 'minutes')
        elapsed_time_hours = (et - st) / (60 * 60)
        print('Total time consumed so far in hours:', elapsed_time_hours, 'hours')

        # Check for early stopping
        if eval['loss'] < best_validation_loss:
            best_validation_loss = eval['loss']
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        # print('early_stopping_counter:', early_stopping_counter)
        print('')

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-c', '--config', default='config/config_fusion.json',type=str,
                        help='Path to the config file (default: config_fusion.json)')

    args = parser.parse_args()

    config = json.load(open(args.config))
    
    main(config)
