import json
import argparse
import torch
import numpy as np
from model.refnetplusplus import refnetplusplus # second publication
from dataset.encoder import ra_encoder
from dataset.dataset_fusion import RADIal
import time
from dataset.dataloader_fusion import CreateDataLoaders

def calculate_fps_fusion_secondpub(model, inputs2, inputs1, is_training):
    start_time = time.time()
    for i in range(100):
        model(inputs2, inputs1, is_training)
    end_time = time.time()
    fps = 100 / (end_time - start_time)
    return fps

def main(config, checkpoint_filename):
    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load dataset and create model
    enc = ra_encoder(geometry=config['dataset']['geometry'],
                     statistics=config['dataset']['statistics'],
                     regression_layer=2)

    dataset = RADIal(config=config,
                     encoder=enc.encode,
                     difficult=True)

    train_loader, val_loader, test_loader = CreateDataLoaders(dataset, config, config['seed'])

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

        print("*******************")
        print("Calculating the FPS!")
        print("*******************")

    net.to(device)

    # Load the model
    dict = torch.load(checkpoint_filename, map_location=device)
    net.load_state_dict(dict['net_state_dict'])
    net.eval()

    fps_list = []
    for idx, data in enumerate(test_loader):
        is_training = False
        inputs1 = data[0].to(device).float()  # radar data
        inputs2 = data[1].to(device).float()  # camera data
        fps = calculate_fps_fusion_secondpub(net, inputs2, inputs1, is_training)
        fps_list.append(fps)
        print(f"FPS for image {idx + 1}: {fps:.2f}")

    average_fps = np.mean(fps_list)
    print("**********************************************")
    print(f"Average FPS for all images: {average_fps:.2f}")

    # Calculate and print the standard deviation of FPS
    std_dev_fps = np.std(fps_list)
    print(f"Standard Deviation of FPS for all images: {std_dev_fps:.2f}")
    print("**********************************************")


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