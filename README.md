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

## How to use this library?
**Important Note: Clone this repository in your project. Please don't add your codes in "learning3d" folder.**

1. All networks are defined in the module "models".
2. All loss functions are defined in the module "losses".
3. Data loaders are pre-defined in data_utils/dataloaders.py file.
4. All pretrained models are provided in learning3d/pretrained folder.

## Documentation
####  Use Point Embedding Networks:
> from learning3d.models import PointNet, DGCNN\
> pn = PointNet(emb_dims=1024, input_shape='bnc', use_bn=False)\
> dgcnn = DGCNN(emb_dims=1024, input_shape='bnc')

emb_dims: &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Size of feature vector for the each point (Integer)\
input_shape: &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Shape of input point cloud. (String) [b: batch, n: no. of points, c: channels]\
output: &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; High dimensional embeddings for each point. [N x embd_dims]

#### Use of Classification Network:
> from learning3d.models import Classifier, PointNet\
> classifier = Classifier(feature_model=PointNet(), num_classes=40)

feature_model: &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Network to encode point clouds in higher dimensional embeddings (Object) [PointNet, DGCNN]\
num_classes: &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; Number of object categories to be classified. (Integer)

### To run codes from examples:
1. Copy the file from "examples" folder outside of the directory "learning3d"
2. Now, run the file. (ex. python test_pointnet.py)
- Your Directory/Location
	- learning3d
	- test_pointnet.py