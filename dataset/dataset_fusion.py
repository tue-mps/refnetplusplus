from torch.utils.data import Dataset
import numpy as np
import os
import cv2
from torchvision.transforms import Resize,CenterCrop
import torchvision.transforms as transform
import pandas as pd
from PIL import Image

class RADIal(Dataset):

    def __init__(self, config, encoder=None,difficult=True):

        self.config = config
        self.encoder = encoder
        root_dir = self.config['dataset']['root_dir']
        self.labels = pd.read_csv(os.path.join(root_dir,'labels.csv')).to_numpy()
       
        # Keeps only easy samples
        # (this means all 1's (difficult samples) in the last column of the csv file are ignored!
        # When difficult=True, then whole dataset is being considered (8252 samples)
        if(difficult==False):
            ids_filters=[]
            ids = np.where(self.labels[:, -1] == 0)[0]
            ids_filters.append(ids)
            ids_filters = np.unique(np.concatenate(ids_filters))
            self.labels = self.labels[ids_filters]

        # Gather each input entries by their sample id
        self.unique_ids = np.unique(self.labels[:,0])
        self.label_dict = {}
        for i,ids in enumerate(self.unique_ids):
            sample_ids = np.where(self.labels[:,0]==ids)[0]
            self.label_dict[ids]=sample_ids
        self.sample_keys = list(self.label_dict.keys())

        self.resize = Resize((256,224), interpolation=transform.InterpolationMode.NEAREST)
        self.crop = CenterCrop((512,448))


    def __len__(self):
        return len(self.label_dict)

    def __getitem__(self, index):

        root_dir = self.config['dataset']['root_dir']
        statistics = self.config['dataset']['statistics']

        # Get the sample id
        sample_id = self.sample_keys[index] 

        # From the sample id, retrieve all the labels ids
        entries_indexes = self.label_dict[sample_id]

        # Get the objects labels
        box_labels = self.labels[entries_indexes]

        # Labels contains following parameters:
        # x1_pix	y1_pix	x2_pix	y2_pix	laser_X_m	laser_Y_m	laser_Z_m radar_X_m	radar_Y_m	radar_R_m

        # format as following [Range, Angle, Doppler,laser_X_m,laser_Y_m,laser_Z_m,x1_pix,y1_pix,x2_pix	,y2_pix, radar_X_m, radar_Y_m]
        box_labels = box_labels[:,[10,11,12,5,6,7,1,2,3,4,8,9]].astype(np.float32)

        # Detection labels
        if (self.encoder != None):
            out_label = self.encoder(box_labels).copy()

        # camera images in perspective view
        cam_img_name = os.path.join(root_dir, 'camera', "image_{:06d}.jpg".format(sample_id))
        cameraimage = cv2.imread(cam_img_name)
        height, width = cameraimage.shape[:2]  # Get original dimensions
        new_size = (width // 2, height // 2)  # Compute new dimensions
        cameraimage_half = cv2.resize(cameraimage, new_size, interpolation=cv2.INTER_AREA)

        # Read the segmentation map in BEV
        segmap_name_polar = os.path.join(root_dir, 'radar_Freespace', "freespace_{:06d}.png".format(sample_id))
        segmap_polar = Image.open(segmap_name_polar)  # [512,900]
        # 512 pix for the range and 900 pix for the horizontal FOV (180deg)
        # We crop the fov to 89.6deg
        segmap_polar = self.crop(segmap_polar)
        # and we resize to half of its size
        segmap_polar = np.asarray(self.resize(segmap_polar)) == 255

        # Read the Radar FFT data
        radar_name = os.path.join(root_dir, 'radar_FFT', "fft_{:06d}.npy".format(sample_id))
        rd_input = np.load(radar_name, allow_pickle=True)
        radar_FFT = np.concatenate([rd_input.real, rd_input.imag], axis=2)
        if (statistics is not None):
            for i in range(len(statistics['input_mean'])):
                radar_FFT[..., i] -= statistics['input_mean'][i]
                radar_FFT[..., i] /= statistics['input_std'][i]

        return radar_FFT, cameraimage_half, segmap_polar, out_label, box_labels, cam_img_name