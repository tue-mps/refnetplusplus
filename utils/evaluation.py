import torch
import numpy as np
from .metrics_bev import Metrics, Metrics_seg, Metrics_det, GetFullMetrics
import pkbar

# outlabels are used to calculate the loss while box labels are used compute accuracy

def run_evaluation(net, loader, device,config,encoder, detection_loss=None, segmentation_loss=None, losses_params=None,mode_params=None):

    if config['model']['DetectionHead'] == 'True' and config['model']['SegmentationHead'] == 'True':
        metrics = Metrics()
        metrics.reset()

    if config['model']['DetectionHead'] == 'False' and config['model']['SegmentationHead'] == 'True':
        metrics_seg = Metrics_seg()
        metrics_seg.reset()

    if config['model']['DetectionHead'] == 'True' and config['model']['SegmentationHead'] == 'False':
        metrics_det = Metrics_det()
        metrics_det.reset()

    net.eval()
    classif_loss = torch.tensor(0, dtype=torch.float64)
    reg_loss = torch.tensor(0, dtype=torch.float64)
    loss_seg = torch.tensor(0, dtype=torch.float64)
    running_loss = 0.0

    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)

    for i, data in enumerate(loader):

        is_training = False
        inputs1 = data[0].to(device).float()  # radar data
        inputs2 = data[1].to(device).float()  # camera half fv image
        seg_map_label = data[2].to(device).double()
        det_label = data[3].to(device).float()
        with torch.set_grad_enabled(False):
            outputs = net(inputs2, inputs1, is_training)

        if (detection_loss != None and segmentation_loss != None):

            if config['model']['DetectionHead'] == 'True':
                classif_loss, reg_loss = detection_loss(outputs['Detection'], det_label, losses_params,mode_params)
                classif_loss *= losses_params['weight'][0]
                reg_loss *= losses_params['weight'][1]

            if config['model']['SegmentationHead'] == 'True':
                prediction = outputs['Segmentation'].contiguous().flatten()
                label = seg_map_label.contiguous().flatten()
                loss_seg = segmentation_loss(prediction, label)
                loss_seg *= inputs1.size(0)
                loss_seg *= losses_params['weight'][2]

            loss = classif_loss + reg_loss + loss_seg
            running_loss += loss.item() * inputs1.size(0)

            if config['model']['DetectionHead'] == 'True' and config['model']['SegmentationHead'] == 'True':
                out_obj = outputs['Detection'].detach().cpu().numpy().copy()
                labels = data[4]
                out_seg = torch.sigmoid(outputs['Segmentation']).detach().cpu().numpy().copy()
                label_freespace = seg_map_label.detach().cpu().numpy().copy()
                for pred_obj, pred_map, true_obj, true_map in zip(out_obj, out_seg, labels, label_freespace):
                    metrics.update(pred_map[0], true_map, np.asarray(encoder.decode(pred_obj, 0.05)), true_obj, threshold=0.2, range_min=5, range_max=100)

            if config['model']['DetectionHead'] == 'False' and config['model']['SegmentationHead'] == 'True':
                out_seg = torch.sigmoid(outputs['Segmentation']).detach().cpu().numpy().copy()
                label_freespace = seg_map_label.detach().cpu().numpy().copy()
                for pred_map, true_map in zip(out_seg, label_freespace):
                    metrics_seg.update(pred_map[0], true_map, threshold=0.2, range_min=5, range_max=100)

            if config['model']['DetectionHead'] == 'True' and config['model']['SegmentationHead'] == 'False':
                out_obj = outputs['Detection'].detach().cpu().numpy().copy()
                labels = data[4]
                for pred_obj, true_obj in zip(out_obj, labels):
                    metrics_det.update(np.asarray(encoder.decode(pred_obj, 0.05)), true_obj, threshold=0.2, range_min=5, range_max=100)

        kbar.update(i)

    if config['model']['DetectionHead'] == 'True' and config['model']['SegmentationHead'] == 'True':
        mAP, mAR, mIoU = metrics.GetMetrics()
        return {'loss': running_loss, 'mAP': mAP, 'mAR': mAR, 'mIoU': mIoU}

    if config['model']['DetectionHead'] == 'False' and config['model']['SegmentationHead'] == 'True':
        mIoU = metrics_seg.GetMetrics()
        return {'loss': running_loss, 'mIoU': mIoU}

    if config['model']['DetectionHead'] == 'True' and config['model']['SegmentationHead'] == 'False':
        mAP, mAR = metrics_det.GetMetrics()
        return {'loss': running_loss, 'mAP': mAP, 'mAR': mAR}


def run_FullEvaluation(net,loader,device,config,encoder):

    net.eval()
    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)
    print('Generating Predictions...')

    predictions = {'prediction':{'objects':[],'freespace':[]},'label':{'objects':[],'freespace':[]}}

    for i, data in enumerate(loader):
        is_training = False
        inputs1 = data[0].to(device).float()  # radar data
        inputs2 = data[1].to(device).float()  # camera half fv image
        seg_map_label = data[2].to(device).double()
        det_label = data[3].to(device).float()
        with torch.set_grad_enabled(False):
            outputs = net(inputs2, inputs1, is_training)

        if config['model']['DetectionHead'] == 'True' and config['model']['SegmentationHead'] == 'True':
            out_obj = outputs['Detection'].detach().cpu().numpy().copy()
            labels = data[4] #box labels [Range, Angle, Doppler,laser_X_m,laser_Y_m,laser_Z_m,x1_pix,y1_pix,x2_pix	,y2_pix, radar_X_m, radar_Y_m]
            out_seg = torch.sigmoid(outputs['Segmentation']).detach().cpu().numpy().copy()
            label_freespace = seg_map_label.detach().cpu().numpy().copy()
            for pred_obj, pred_map, true_obj, true_map in zip(out_obj, out_seg, labels, label_freespace):
                predictions['prediction']['objects'].append(np.asarray(encoder.decode(pred_obj, 0.05)))
                predictions['label']['objects'].append(true_obj)
                predictions['prediction']['freespace'].append(pred_map[0])
                predictions['label']['freespace'].append(true_map)
            kbar.update(i)

        if config['model']['DetectionHead'] == 'False' and config['model']['SegmentationHead'] == 'True':
            out_seg = torch.sigmoid(outputs['Segmentation']).detach().cpu().numpy().copy()
            label_freespace = seg_map_label.detach().cpu().numpy().copy()
            for pred_map, true_map in zip(out_seg, label_freespace):
                predictions['prediction']['freespace'].append(pred_map[0])
                predictions['label']['freespace'].append(true_map)
            kbar.update(i)

        if config['model']['DetectionHead'] == 'True' and config['model']['SegmentationHead'] == 'False':
            out_obj = outputs['Detection'].detach().cpu().numpy().copy()
            labels = data[4]
            for pred_obj, true_obj in zip(out_obj, labels):
                predictions['prediction']['objects'].append(np.asarray(encoder.decode(pred_obj, 0.05)))
                predictions['label']['objects'].append(true_obj)
            kbar.update(i)

    if config['model']['DetectionHead'] == 'True':
        GetFullMetrics(predictions['prediction']['objects'],
                       predictions['label']['objects'],
                       range_min=5, range_max=100, IOU_threshold=0.5)

    if (config['model']['SegmentationHead'] == 'True'):
        mIoU = []
        for i in range(len(predictions['prediction']['freespace'])):
            # 0 to 124 means 0 to 50m
            pred = predictions['prediction']['freespace'][i][:124].reshape(-1) >= 0.5
            label = predictions['label']['freespace'][i][:124].reshape(-1)
            intersection = np.abs(pred * label).sum()
            union = np.sum(label) + np.sum(pred) - intersection
            iou = intersection / union
            mIoU.append(iou)
        mIoU = np.asarray(mIoU).mean()
        print('------- Freespace Scores ------------')
        print('  mIoU', mIoU * 100, '%')


