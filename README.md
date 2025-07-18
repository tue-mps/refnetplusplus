# REFNet++: Multi-Task Efficient Fusion of Camera and Radar Sensor Data in Bird’s-Eye Polar View

**[Kavin Chandrasekaran](https://scholar.google.com/citations?user=FMeH0ZkAAAAJ&hl=en)<sup>1,2</sup>, [Sorin Grigorescu](https://scholar.google.com/citations?user=3TsU0iMAAAAJ&hl=en)<sup>1,3</sup>, [Gijs Dubbelman](https://scholar.google.nl/citations?user=wy57br8AAAAJ)<sup>2</sup>, [Pavol Jancura](https://scholar.google.com/citations?user=ApILewUAAAAJ&hl=en)<sup>2</sup>**

¹ Elektrobit Automotive GmbH  
² Eindhoven University of Technology  
³ Transilvania University of Brasov

## News
- **(2025/07/01)** Accepted to IEEE ITSC 2025!

## Overview
The variational encoder-decoder architecture learns the transformation from the front-view camera image to BEV, which corresponds to the Range-Azimuth (RA) domain. 

On the other hand, the radar encoder-decoder architecture learns to recover the angle information from the complex Range-Doppler (RD) input, producing RA features. 

Free space segmentation and vehicle detection are performed on the resulting features fused by concatenation by the appropriate heads. (a) ground truth labels, (b) prediction results, and (c) predictions projected onto the camera image.

<p align="center">
  <img src="images/overview.png" div align=center>
</p> 

## Abstract
A realistic view of the vehicle’s surroundings is generally offered by camera sensors, which is crucial for environmental perception. Affordable radar sensors, on the other hand, are becoming invaluable due to their robustness in variable weather conditions. However, because of their noisy output and reduced classification capability, they work best when combined with other sensor data. Specifically, we address the challenge of multimodal sensor fusion by aligning radar and camera data in a unified domain, prioritizing not only accuracy, but also computational efficiency. Our work leverages the raw range-Doppler (RD) spectrum from radar
and front-view camera images as inputs. To enable effective fusion, we employ a variational encoder-decoder architecture that learns the transformation of front-view camera data into the Bird’s-Eye View (BEV) polar domain. Concurrently, a radar encoder-decoder learns to recover the angle information from the RD data that produce Range-Azimuth (RA) features. This alignment ensures that both modalities are represented in a compatible domain, facilitating robust and efficient sensor fusion. This work is an enhanced version of our previous architecture [REFNet](https://github.com/tue-mps/refnet). We evaluated our fusion strategy for vehicle detection and free space segmentation against state-of-the-art methods using the RADIal dataset.

## Fusion Architecture
The input to the radar only network is the range-Doppler (RD) data, while the camera only network intakes front-view camera images. x0 to x4 are the feature maps from the respective encoder blocks. The encoder is connected to the decoder by a thick blue arrow, by which the encoded features are upscaled to higher resolutions. The skip connections are shown as dotted lines that preserves the spatial information. The radar and camera features are fused by concatenation on the subsequent heads. Predictions are in Bird’s Eye RA Polar View.

<p align="center">
  <img src="images/detailed_arch.png" div align=center>
</p>

The models are trained and tested on the [RADIal dataset](https://github.com/valeoai/RADIal/tree/main). The dataset can be downloaded
[here](https://github.com/valeoai/RADIal/tree/main#labels:~:text=Download%20instructions). Our model is located under `pretrainedmodel/refnetplusplus.pth`.

## Model Predictions
Qualitative results on samples from the test set. (a) depicts the ground truth labels in Bird’s-Eye Polar View where the free space segmentation is in white while the BEV bounding boxes are in green. (b) are our prediction results. For a better visualization, the bounding box predictions are projected onto the camera images as shown in (c), with the ground-truth boxes. The free space predictions are projected by consolidating the intensity values.

<p align="center">
  <img src="images/qualitative.png" div align=center>
</p>

## Setting up the virtual environment
### Requirements
All the codes are tested in the following environment:
- Linux (tested on Ubuntu 22.04)
- Python 3.9

### Installation
0. Install conda if you don't have it installed. Then clone the repo and set up the conda environment:
```bash
$ git clone "this repo"
$ conda create --prefix "your_path" python=3.9 -y
$ conda update -n base -c defaults conda
$ conda activate "your_path"
```

1. The following are the packages used:
```bash
$ conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
$ pip install -U pip
$ pip3 install pkbar
$ pip3 install tensorboard
$ pip3 install pandas
$ pip3 install shapely
$ pip3 install jupyter
$ pip3 install pickleshare
$ pip3 install opencv-python
$ pip3 install einops
$ pip3 install timm
$ pip3 install scipy
$ pip3 install scikit-learn
$ pip3 install polarTransform
$ pip3 install matplotlib
$ pip3 install numpy==1.23
```
## Running the code

### Training
Run the following to train the model from scratch. Please check the `config/config_fusion.json` before training. REFNet++ in multi-tasking mode will be chosen by default. To train only one task, then set the `DetectionHead` or `SegmentationHead` to `True` appropriately.

```bash
$ python 1-Train.py
```

### Evaluation
To evaluate the model performance, please load the trained model and run:
```bash
$ python 2-Evaluation.py
```

### Testing
To obtain qualitative results, please load the trained model and run:
```bash
$ python 3-Test.py
```
A video like this should pop up, where gt &rarr; ground truth; pred &rarr; our model predictions in multi-tasking mode; overlay &rarr; predictions are projected on to the camera frames:

<p align="center">
  <img src="images/refnetplusplus.gif" div align=center>
</p>


### Computational complexity
Frames Per Second (FPS) is measured on NVIDIA RTX A6000. To compute FPS, please load the trained model and run:
```bash
$ python 4-FPS.py
```

## Conclusion and Further research
- We proposed REFNet++, a fusion architecture that performs multitasking and can also operate in a single task mode, designed to boost the computational efficiency of the camera-radar perception system. 
- In line with our research goal and the results demonstrated on the RADIal dataset, our method exhibits excellent trade-off between performance while retaining a comparatively low computing power.
- Our code can be extended for further analysis, for example to include LiDAR data.
- We plan to further accelerate this research using other fusion datasets.

## Acknowledgments
- Thanks to Elektrobit Automotive GmbH and Mobile Perception Systems Lab from Eindhoven University of Technology for their continous support.
- Visit our previous work: [REFNet](https://github.com/tue-mps/refnet/).

## License
The repo is released under the BSD 3-Clause License.
