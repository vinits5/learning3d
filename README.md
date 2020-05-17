# Learning3D: A Modern Library for Deep Learning on 3D Point Clouds Data.

**[Documentation]() | [Quick Start]() | [Python]() | [Demo](https://github.com/vinits5/learning3d/blob/master/examples/test_pointnet.py)**

Learning3D is an open-source library that supports the development of deep learning algorithms that deal with 3D data. The Learning3D exposes a set of state of art deep neural networks in python. A modular code has been provided for further development. We welcome contributions from the open-source community.

## Available Computer Vision Algorithms in Learning3D

| Sr. No.       | Tasks         | Algorithms  |
|:-------------:|:----------:|:-----|
| 1 | Classification | PointNet, DGCNN |
| 2 | Segmentation | PointNet, DGCNN |
| 3 | Reconstruction | Point Completion Network (PCN) |
| 4 | Registration | PointNetLK, PCRNet, DCP, PRNet |
| 5 | Flow Estimation | FlowNet3D | 

## Available Pretrained Models
1. PointNet
2. PCN
3. PointNetLK
4. PCRNet
5. DCP
6. PRNet
7. FlowNet3D

## Available Datasets
1. ModelNet40

## Technical Details
### Supported OS
1. Ubuntu 16.04
2. Ubuntu 18.04
3. Linux Mint

### Requirements
1. CUDA 10.0 or higher
2. Pytorch 1.3 or higher
3. Transforms3d 0.3 or higher
4. Ninja
5. H5py

### To run codes from examples:
1. Copy the file from "examples" folder outside of the directory "learning3d"
2. Now, run the file. (ex. python test_pointnet.py)\

Your directory should look like this:
- Your Directory/Location
	- learning3d
	- test_pointnet.py