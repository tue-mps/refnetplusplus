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
import re

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

    # Set up the VideoWriter
    save_video = True
    video_1 = cv2.VideoWriter(f'/home/kach271771/refnetplusplus_fv.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 10, (850, 540))
    video_2 = cv2.VideoWriter(f'/home/kach271771/refnetplusplus_bev.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 10, (448, 256))

    for data in dataset: #this considers the full dataset, you can also check it using the "test_loader"
        is_training = False
        inputs1 = torch.tensor(data[0]).permute(2, 0, 1).to(device).float().unsqueeze(0)
        inputs2 = torch.tensor(data[1]).permute(2, 0, 1).to(device).float().unsqueeze(0)
        seg_map_label = torch.tensor(data[2]).to(device).double().unsqueeze(0)
        det_label = torch.tensor(data[3]).to(device).float().unsqueeze(0)
        box_labels = data[4]
        sample_id = re.search(r'_([0-9]+)\.jpg$', data[5])
        sample_id = sample_id.group(1)
        sample_id = int(sample_id)
        with torch.set_grad_enabled(False):
            outputs = net(inputs2, inputs1, is_training)

        (seg_labels_flip, out_seg_flip,
         modeloutput_seg, overlay) = DisplayHMI(data[5], inputs1,
                                               seg_map_label, box_labels,
                                               outputs, enc, sample_id,
                                               datapath=config['dataset']['root_dir'])

        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        out = np.hstack((seg_labels_flip, out_seg_flip))

        if save_video == True:
            overlay = overlay.astype(np.float32) * 255.0
            overlay = overlay.astype(np.uint8)
            out = out.astype(np.float32) * 255.0
            out = out.astype(np.uint8)
            video_1.write(overlay)
            video_2.write(out)
            cv2.imshow('REFNet++', overlay)
            cv2.imshow('Prediction Vs Ground-Truth', out)
        else:
            cv2.imshow('REFNet++', overlay)
            cv2.imshow('Ground-Truth Vs Prediction in Polar Domain', out)
            # cv2.waitKey(0) # if you want to visualize frame by frame slowly, then uncomment this line (to go to next frame press space bar key)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    video_1.release()
    video_2.release()
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