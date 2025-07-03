import json
import argparse
import torch
import numpy as np
from model.refnetplusplus import refnetplusplus
from dataset.encoder import ra_encoder
from dataset.dataset_fusion import RADIal
from dataset.dataloader_fusion import CreateDataLoaders
import cv2
from utils.util import DisplayHMI

gpu_id = 0

def main(config, checkpoint_filename):

    # set device
    device = torch.device('cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')
    print("Device used:", device)

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

    net.to(device)

    # Load the model
    dict = torch.load(checkpoint_filename, map_location=device)
    net.load_state_dict(dict['net_state_dict'])
    net.eval()

    for data in test_loader:
        is_training = False
        inputs1 = data[0].to(device).float()  # radar data
        inputs2 = data[1].to(device).float()  # camera half fv image
        seg_map_label = data[2].to(device).double()
        det_label = data[3].to(device).float()
        with torch.set_grad_enabled(False):
            outputs = net(inputs2, inputs1, is_training)

        hmi = DisplayHMI(data[3], seg_map_label, det_label,outputs)

        cv2.imshow('Multi-Tasking',hmi)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='REFNet++ Predictions')
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