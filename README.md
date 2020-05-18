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
B: Batch Size, N: No. of points and C: Channels.
####  Use of Point Embedding Networks:
> from learning3d.models import PointNet, DGCNN\
> pn = PointNet(emb_dims=1024, input_shape='bnc', use_bn=False)\
> dgcnn = DGCNN(emb_dims=1024, input_shape='bnc')

| Sr. No. | Variable | Data type | Shape | Choices | Use |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1. | emb_dims | Integer | Scalar | 1024, 512 | Size of feature vector for the each point|
| 2. | input_shape | String | - | 'bnc', 'bcn' | Shape of input point cloud|
| 3. | output | tensor | BxCxN | - | High dimensional embeddings for each point|

#### Use of Classification / Segmentation Network:
> from learning3d.models import Classifier, PointNet, Segmentation\
> classifier = Classifier(feature_model=PointNet(), num_classes=40)\
> seg = Segmentation(feature_model=PointNet(), num_classes=40)

| Sr. No. | Variable | Data type | Shape | Choices | Use |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1. | feature_model | Object | - | PointNet / DGCNN | Point cloud embedding network |
| 2. | num_classes | Integer | Scalar | 10, 40 | Number of object categories to be classified |
| 3. | output | tensor | Classification: Bx40, Segmentation: BxNx40 | 10, 40 | Probabilities of each category or each point |

#### Use of Registration Networks:
> from learning3d.models import PointNet, PointNetLK, DCP, iPCRNet, PRNet\
> pnlk = PointNetLK(feature_model=PointNet(), delta=1e-02, xtol=1e-07, p0_zero_mean=True, p1_zero_mean=True, pooling='max')\
> dcp = DCP(feature_model=PointNet(), pointer_='transformer', head='svd')\
> pcrnet = iPCRNet(feature_moodel=PointNet(), pooling='max')

| Sr. No. | Variable | Data type | Choices | Use | Algorithm |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1. | feature_model | Object | PointNet / DGCNN | Point cloud embedding network | PointNetLK | 
| 2. | delta | Float | Scalar | Parameter to calculate approximate jacobian | PointNetLK |
| 3. | xtol | Float | Scalar | Check tolerance to stop iterations | PointNetLK |
| 4. | p0_zero_mean | Boolean | True/False | Subtract mean from template point cloud |  PointNetLK |
| 5. | p1_zero_mean | Boolean | True/False | Subtract mean from source point cloud | PointNetLK |
| 6. | pooling | String | 'max' / 'avg' | Type of pooling used to get global feature vectror | PointNetLK |
| 7. | pointer_ | String | 'transformer' / 'identity' | Choice for Transformer/Attention network | DCP |
| 8. | head | String | 'svd' / 'mlp' | Choice of module to estimate registration params | DCP |

#### Use of Point Completion Network:
> from learning3d.models import PCN\
> pcn = PCN(emb_dims=1024, input_shape='bnc', num_coarse=1024, grid_size=4, detailed_output=True)

| Sr. No. | Variable | Data type | Choices | Use |
|:---:|:---:|:---:|:---:|:---:|
| 1. | emb_dims | Integer | 1024, 512 | Size of feature vector for each point | 
| 2. | input_shape | String | 'bnc' / 'bcn' | Shape of input point cloud |
| 3. | num_coarse | Integer | 1024 | Shape of output point cloud |
| 4. | grid_size | Integer | 4, 8, 16 | Size of grid used to produce detailed output | 
| 5. | detailed_output | Boolean | True / False | Choice for additional module to create detailed output point cloud|

#### Use of Flow Estimation Network:
> from learning3d.models import FlowNet3D\
> flownet = FlowNet3D()

### To run codes from examples:
1. Copy the file from "examples" folder outside of the directory "learning3d"
2. Now, run the file. (ex. python test_pointnet.py)
- Your Directory/Location
	- learning3d
	- test_pointnet.py