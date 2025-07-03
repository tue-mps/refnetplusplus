import torch
import numpy as np
import cv2
from shapely.geometry import Polygon
import polarTransform
import os
from PIL import Image

# Camera parameters
camera_matrix = np.array([[1.84541929e+03, 0.00000000e+00, 8.55802458e+02],
                          [0.00000000e+00, 1.78869210e+03, 6.07342667e+02], [0., 0., 1]])
dist_coeffs = np.array([2.51771602e-01, -1.32561698e+01, 4.33607564e-03, -6.94637533e-03, 5.95513933e+01])
rvecs = np.array([1.61803058, 0.03365624, -0.04003127])
tvecs = np.array([0.09138029, 1.38369885, 1.43674736])
ImageWidth = 1920
ImageHeight = 1080

AoA_mat = np.load('/home/kach271771/Music/refnetplusplus/dataset/CalibrationTable.npy',allow_pickle=True).item()

numSamplePerChirp = 512
numRxPerChip = 4
numChirps = 256
numRxAnt = 16
numTxAnt = 12
numReducedDoppler = 16
numChirpsPerLoop = 16
dividend_constant_arr = np.arange(0, numReducedDoppler*numChirpsPerLoop ,numReducedDoppler)
window = np.array(AoA_mat['H'][0])
CalibMat=AoA_mat['Signal'][...,5]

def worldToImage(x, y, z):
    world_points = np.array([[x, y, z]], dtype='float32')
    rotation_matrix = cv2.Rodrigues(rvecs)[0]

    imgpts, _ = cv2.projectPoints(world_points, rotation_matrix, tvecs, camera_matrix, dist_coeffs)

    u = int(min(max(0, imgpts[0][0][0]), ImageWidth - 1))
    v = int(min(max(0, imgpts[0][0][1]), ImageHeight - 1))

    return u, v


def RA_to_cartesian_box(data):
    L = 4
    W = 1.8

    boxes = []
    for i in range(len(data)):
        x = np.sin(np.radians(data[i][1])) * data[i][0]
        y = np.cos(np.radians(data[i][1])) * data[i][0]

        boxes.append([x - W / 2, y, x + W / 2, y, x + W / 2, y + L, x - W / 2, y + L, data[i][0], data[i][1]])

    return boxes


def perform_nms(valid_class_predictions, valid_box_predictions, nms_threshold):
    # sort the detections such that the entry with the maximum confidence score is at the top
    sorted_indices = np.argsort(valid_class_predictions)[::-1]
    sorted_box_predictions = valid_box_predictions[sorted_indices]
    sorted_class_predictions = valid_class_predictions[sorted_indices]

    for i in range(sorted_box_predictions.shape[0]):
        # get the IOUs of all boxes with the currently most certain bounding box
        try:
            ious = np.zeros((sorted_box_predictions.shape[0]))
            ious[i + 1:] = bbox_iou(sorted_box_predictions[i, :8], sorted_box_predictions[i + 1:, :8])
        except ValueError:
            break
        except IndexError:
            break

        # eliminate all detections which have IoU > threshold
        overlap_mask = np.where(ious < nms_threshold, True, False)
        sorted_box_predictions = sorted_box_predictions[overlap_mask]
        sorted_class_predictions = sorted_class_predictions[overlap_mask]

    return sorted_class_predictions, sorted_box_predictions


def bbox_iou(box1, boxes):
    # currently inspected box
    box1 = box1.reshape((4, 2))
    rect_1 = Polygon([(box1[0, 0], box1[0, 1]), (box1[1, 0], box1[1, 1]), (box1[2, 0], box1[2, 1]),
                      (box1[3, 0], box1[3, 1])])
    area_1 = rect_1.area

    # IoU of box1 with each of the boxes in "boxes"
    ious = np.zeros(boxes.shape[0])
    for box_id in range(boxes.shape[0]):
        box2 = boxes[box_id]
        box2 = box2.reshape((4, 2))
        rect_2 = Polygon([(box2[0, 0], box2[0, 1]), (box2[1, 0], box2[1, 1]), (box2[2, 0], box2[2, 1]),
                          (box2[3, 0], box2[3, 1])])
        area_2 = rect_2.area

        # get intersection of both bounding boxes
        inter_area = rect_1.intersection(rect_2).area

        # compute IoU of the two bounding boxes
        iou = inter_area / (area_1 + area_2 - inter_area)

        ious[box_id] = iou

    return ious


def process_predictions_FFT(batch_predictions, confidence_threshold=0.1, nms_threshold=0.05):
    point_cloud_reg_predictions = RA_to_cartesian_box(batch_predictions)
    point_cloud_reg_predictions = np.asarray(point_cloud_reg_predictions)
    point_cloud_class_predictions = batch_predictions[:, -1]

    # get valid detections
    validity_mask = np.where(point_cloud_class_predictions > confidence_threshold, True, False)
    valid_box_predictions = point_cloud_reg_predictions[validity_mask]
    valid_class_predictions = point_cloud_class_predictions[validity_mask]

    # perform Non-Maximum Suppression
    final_class_predictions, final_box_predictions = perform_nms(valid_class_predictions, valid_box_predictions,
                                                                 nms_threshold)

    # concatenate point_cloud_id, confidence score and bounding box prediction | shape: [N_FINAL, 1+1+8]
    final_point_cloud_predictions = np.hstack((final_class_predictions[:, np.newaxis],
                                               final_box_predictions))

    return final_point_cloud_predictions


def DisplayHMI(input_path, rd_spectrum, labels_Seg, box_labels, model_outputs, enc, sample_id, datapath):
    image = np.asarray(Image.open(input_path))
    image_ = image.copy()
    cam_freespace_path = os.path.join(datapath, 'camera_Freespace', "freespace_{:06d}.png".format(sample_id))
    cam_freespace = cv2.imread(cam_freespace_path)
    crop_image_ = image_[:, :850, :]
    image_ = cv2.addWeighted(crop_image_, 0.7, cam_freespace, 0.3, 0)

    # Model outputs
    pred_obj_ = model_outputs['Detection'].detach().cpu().numpy().copy()[0]
    out_seg = torch.sigmoid(model_outputs['Segmentation']).detach().cpu().numpy().copy()[0, 0]

    # Decode the output detection map
    pred_obj_rac = enc.decode(pred_obj_, 0.05)
    pred_obj = np.asarray(pred_obj_rac)

    # process prediction: polar to cartesian, NMS...
    if (len(pred_obj) > 0):
        pred_obj = process_predictions_FFT(pred_obj, confidence_threshold=0.2)

    ## FFT
    rd_spectrum = rd_spectrum.squeeze(0).permute(1, 2, 0)
    rd_spectrum = rd_spectrum.cpu().numpy()
    FFT = np.abs(rd_spectrum[..., :16] + rd_spectrum[..., 16:] * 1j).mean(axis=2)
    PowerSpectrum = np.log10(FFT)
    # rescale
    PowerSpectrum = (PowerSpectrum - PowerSpectrum.min()) / (PowerSpectrum.max() - PowerSpectrum.min()) * 255
    PowerSpectrum = cv2.cvtColor(PowerSpectrum.astype('uint8'), cv2.COLOR_GRAY2BGR)

    object_pred_pic = []
    seg_labels = labels_Seg.squeeze(0).cpu().numpy()
    seg_labels = seg_labels.astype(np.float32)
    seg_labels = cv2.flip(seg_labels, flipCode=1)
    seg_labels = cv2.merge((seg_labels, seg_labels, seg_labels))
    seg_labels = (seg_labels * 255).astype(np.uint8)

    RA_cartesian, _ = polarTransform.convertToCartesianImage(np.moveaxis(out_seg, 0, 1), useMultiThreading=True,
                                                             initialAngle=0, finalAngle=np.pi, order=0, hasColor=False)

    # Make a crop on the angle axis
    RA_cartesian = RA_cartesian[:, 256 - 100:256 + 100]
    RA_cartesian = np.asarray((RA_cartesian * 255).astype('uint8'))
    RA_cartesian = cv2.cvtColor(RA_cartesian, cv2.COLOR_GRAY2BGR)
    RA_cartesian = cv2.resize(RA_cartesian, dsize=(400, 512))
    RA_cartesian = cv2.flip(RA_cartesian, flipCode=-1)

    out_seg = cv2.flip(out_seg, flipCode=1)
    out_seg_rgb_ = cv2.merge((out_seg, out_seg, out_seg))
    out_seg_rgb_ = out_seg_rgb_.astype(np.float32)
    out_seg_rgb = (out_seg_rgb_ * 255).astype(np.uint8)

    width = 10
    height = 20
    vertical_offset = 10
    for box_label in box_labels: #ground truth bbox
        x1, y1, x2, y2 = box_label[6], box_label[7], box_label[8], box_label[9]
        # Convert to integer and ensure coordinates are within image dimensions
        x1, y1, x2, y2 = int(x1 / 2), int(y1 / 2), int(x2 / 2), int(y2 / 2)
        image_ = cv2.rectangle(image_, (x1, y1), (x2, y2), (118,230,0), 2)
        Range = box_label[0] * 512 / 103 / 2
        Azimuth = -box_label[1] / 0.4
        Azimuth = seg_labels.shape[1] / 2 - Azimuth
        top_left = (int(Azimuth) - width // 2, int(Range) - height // 2 + vertical_offset)
        bottom_right = (int(Azimuth) + width // 2, int(Range) + height // 2 + vertical_offset)
        cv2.rectangle(seg_labels, top_left, bottom_right, color=(118,230,0), thickness=2)  # use thickness=-1 for Filled red rectangle

    for box in pred_obj:
        box = box[1:]
        u1, v1 = worldToImage(-box[2], box[1], 0)
        u2, v2 = worldToImage(-box[0], box[1], 1.6)
        u1 = int(u1 / 2)
        v1 = int(v1 / 2)
        u2 = int(u2 / 2)
        v2 = int(v2 / 2)

        finall_pred_pic = [(u1, v1), (u2, v2)]
        object_pred_pic.append(finall_pred_pic)

        image_ = cv2.rectangle(image_, (u1, v1), (u2, v2), (255,110,64), 2)

        R = pred_obj[:, 9]
        A = pred_obj[:, 10]
        Range = R * 512 / 103 / 2
        Azimuth = -A / 0.4
        Azimuth = out_seg.shape[1] / 2 - Azimuth
        coords = list(zip(Azimuth, Range))
        for a, r in coords:
            top_left = (int(a) - width // 2, int(r) - height // 2 + vertical_offset)
            bottom_right = (int(a) + width // 2, int(r) + height // 2 + vertical_offset)
            cv2.rectangle(out_seg_rgb, top_left, bottom_right, color=(64,110,255), thickness=2)  # use thickness=-1 for Filled red rectangle

    standard_out = np.hstack((PowerSpectrum, RA_cartesian))
    out_seg_rgb = out_seg_rgb.astype(np.float32) / 255.0
    out_seg_flip = cv2.flip(out_seg_rgb, flipCode=-1)
    seg_labels = seg_labels.astype(np.float32) / 255.0
    seg_labels_flip = cv2.flip(seg_labels, flipCode=-1)
    out_seg_final = np.hstack((seg_labels_flip, out_seg_flip))

    out_seg_rgb_ = out_seg_rgb_.astype(np.float32)
    out_seg_rgb_ = cv2.flip(out_seg_rgb_, flipCode=-1)

    return seg_labels_flip, out_seg_flip, out_seg_rgb_, image_.astype(np.float32) / 255.0

