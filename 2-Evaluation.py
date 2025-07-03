import json
import argparse
import torch
import random
import numpy as np
from model.refnetplusplus import refnetplusplus # second publication
from dataset.encoder import ra_encoder
from dataset.dataset_fusion import RADIal
from dataset.dataloader_fusion import CreateDataLoaders
from utils.evaluation import run_FullEvaluation

def main(config, checkpoint):

    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load dataset and create model
    if config['model']['view_birdseye'] == 'True':
        enc = ra_encoder(geometry=config['dataset']['geometry'],
                         statistics=config['dataset']['statistics'],
                         regression_layer=2)

        dataset = RADIal(config=config,
                         encoder=enc.encode,
                         difficult=True)

        train_loader, val_loader, test_loader = CreateDataLoaders(dataset, config, config['seed'])

        if config['architecture']['bev']['refnetplusplus'] == 'True':

            net = refnetplusplus(mimo_layer=config['model']['MIMO_output'],
                              channels=config['model']['channels'],
                              channels_bev=config['model']['channels_bev'],
                              blocks=config['model']['backbone_block'],
                              detection_head=config['model']['DetectionHead'],
                              segmentation_head=config['model']['SegmentationHead'],
                              config=config, regression_layer=2)

        print("*******************")
        print("Evaluating REFNet++")
        print("*******************")


    net.to(device)

    print('===========  Loading the model ==================:')
    dict = torch.load(checkpoint, map_location=device)
    net.load_state_dict(dict['net_state_dict'])
    
    print('===========  Running the evaluation ==================:')
    run_FullEvaluation(net=net, loader=test_loader,
                       device=device, config=config,
                       encoder=enc)

if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='REFNet++ Evaluation')
    parser.add_argument('-c', '--config',
                        default='/pretrainedmodel/config_fusion.json',
                        type=str,
                        help='Path to the config file (default: config_fusion.json)')
    parser.add_argument('-r', '--checkpoint',
                        default="/pretrainedmodel/refnetplusplus.pth",
                        type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('--difficult', action='store_true')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config, args.checkpoint)