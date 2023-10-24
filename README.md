<p align="center">
	<img src="https://github.com/vinits5/learning3d/blob/master/images/logo.png" height="170">
</p>

# Learning3D: A Modern Library for Deep Learning on 3D Point Clouds Data.

**[Documentation](https://github.com/vinits5/learning3d#documentation) | [Blog](https://medium.com/@vinitsarode5/learning3d-a-modern-library-for-deep-learning-on-3d-point-clouds-data-48adc1fd3e0?sk=0beb59651e5ce980243bcdfbf0859b7a) | [Demo](https://github.com/vinits5/learning3d/blob/master/examples/test_pointnet.py)**

Learning3D is an open-source library that supports the development of deep learning algorithms that deal with 3D data. The Learning3D exposes a set of state of art deep neural networks in python. A modular code has been provided for further development. We welcome contributions from the open-source community.

## Latest News:
1. \[24 Oct, 2023\]: [MaskNet++](https://github.com/zhouruqin/MaskNet2) is now a part of learning3d library.
2. \[12 May, 2022\]: [ChamferDistance](https://github.com/fwilliams/fml) loss function is incorporated in learning3d. This is a purely pytorch based loss function.
3. \[24 Dec. 2020\]: [MaskNet](https://arxiv.org/pdf/2010.09185.pdf) is now ready to enhance the performance of registration algorithms in learning3d for occluded point clouds.
4. \[24 Dec. 2020\]: Loss based on the predicted and ground truth correspondences is added in learning3d after consideration of [Correspondence Matrices are Underrated](https://arxiv.org/pdf/2010.16085.pdf) paper.
5. \[24 Dec. 2020\]: [PointConv](https://arxiv.org/abs/1811.07246), latent feature estimation using convolutions on point clouds is now available in learning3d.
6. \[16 Oct. 2020\]: [DeepGMR](https://wentaoyuan.github.io/deepgmr/), registration using gaussian mixture models is now available in learning3d
7. \[14 Oct. 2020\]: Now, use your own data in learning3d. (Check out [UserData](https://github.com/vinits5/learning3d#use-your-own-data) functionality!)

## Available Computer Vision Algorithms in Learning3D

| Sr. No.       | Tasks         | Algorithms  |
|:-------------:|:----------:|:-----|
| 1 | [Classification](https://github.com/vinits5/learning3d#use-of-classification--segmentation-network) | PointNet, DGCNN, PPFNet, [PointConv](https://github.com/vinits5/learning3d#use-of-pointconv) |
| 2 | [Segmentation](https://github.com/vinits5/learning3d#use-of-classification--segmentation-network) | PointNet, DGCNN |
| 3 | [Reconstruction](https://github.com/vinits5/learning3d#use-of-point-completion-network) | Point Completion Network (PCN) |
| 4 | [Registration](https://github.com/vinits5/learning3d#use-of-registration-networks) | PointNetLK, PCRNet, DCP, PRNet, RPM-Net, DeepGMR |
| 5 | [Flow Estimation](https://github.com/vinits5/learning3d#use-of-flow-estimation-network) | FlowNet3D |
| 6 | [Inlier Estimation](https://github.com/vinits5/learning3d#use-of-inlier-estimation-network-masknet) | MaskNet, MaskNet++ | 

## Available Pretrained Models
1. PointNet
2. PCN
3. PointNetLK
4. PCRNet
5. DCP
6. PRNet
7. FlowNet3D
8. RPM-Net (clean-trained.pth, noisy-trained.pth, partial-pretrained.pth)
9. DeepGMR
10. PointConv (Download from this [link](https://github.com/DylanWusee/pointconv_pytorch/blob/master/checkpoints/checkpoint.pth))
11. MaskNet
12. MaskNet++ / MaskNet2

## Available Datasets
1. ModelNet40

## Available Loss Functions
1. Classification Loss (Cross Entropy)
2. Registration Losses (FrobeniusNormLoss, RMSEFeaturesLoss)
3. Distance Losses (Chamfer Distance, Earth Mover's Distance)
4. Correspondence Loss (based on this [paper](https://arxiv.org/pdf/2010.16085.pdf))

## Technical Details
### Supported OS
1. Ubuntu 16.04
2. Ubuntu 18.04
3. Ubuntu 20.04.6
3. Linux Mint

### Requirements
1. CUDA 10.0 or higher
2. Pytorch 1.3 or higher
3. Python 3.8

## How to use this library?
**Important Note: Clone this repository in your project. Please don't add your codes in "learning3d" folder.**

1. All networks are defined in the module "models".
2. All loss functions are defined in the module "losses".
3. Data loaders are pre-defined in data_utils/dataloaders.py file.
4. All pretrained models are provided in learning3d/pretrained folder.

## Documentation
B: Batch Size, N: No. of points and C: Channels.
####  Use of Point Embedding Networks:
> from learning3d.models import PointNet, DGCNN, PPFNet\
> pn = PointNet(emb_dims=1024, input_shape='bnc', use_bn=False)\
> dgcnn = DGCNN(emb_dims=1024, input_shape='bnc')\
> ppf = PPFNet(features=['ppf', 'dxyz', 'xyz'], emb_dims=96, radius='0.3', num_neighbours=64)

| Sr. No. | Variable | Data type | Shape | Choices | Use |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1. | emb_dims | Integer | Scalar | 1024, 512 | Size of feature vector for the each point|
| 2. | input_shape | String | - | 'bnc', 'bcn' | Shape of input point cloud|
| 3. | output | tensor | BxCxN | - | High dimensional embeddings for each point|
| 4. | features | List of Strings | - | ['ppf', 'dxyz', 'xyz'] | Use of various features |
| 5. | radius | Float | Scalar | 0.3 | Radius of cluster for computing local features |
| 6. | num_neighbours | Integer | Scalar | 64 | Maximum number of points to consider per cluster |

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
> from learning3d.models import PointNet, PointNetLK, DCP, iPCRNet, PRNet, PPFNet, RPMNet\
> pnlk = PointNetLK(feature_model=PointNet(), delta=1e-02, xtol=1e-07, p0_zero_mean=True, p1_zero_mean=True, pooling='max')\
> dcp = DCP(feature_model=PointNet(), pointer_='transformer', head='svd')\
> pcrnet = iPCRNet(feature_moodel=PointNet(), pooling='max')\
> rpmnet = RPMNet(feature_model=PPFNet())\
> deepgmr = DeepGMR(use_rri=True, feature_model=PointNet(), nearest_neighbors=20)

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
| 9. | use_rri | Boolean | True/False | Use nearest neighbors to estimate point cloud features. | DeepGMR |
| 10. | nearest_neighbores | Integer | 20/any integer | Give number of nearest neighbors used to estimate features | DeepGMR |

#### Use of Inlier Estimation Network (MaskNet):
> from learning3d.models import MaskNet, PointNet, MaskNet2\
> masknet = MaskNet(feature_model=PointNet(), is_training=True)
> masknet2 = MaskNet2(feature_model=PointNet(), is_training=True)

| Sr. No. | Variable | Data type | Choices | Use |
|:---:|:---:|:---:|:---:|:---:|
| 1. | feature_model | Object | PointNet / DGCNN | Point cloud embedding network |
| 2. | is_training | Boolean | True / False | Specify if the network will undergo training or testing |

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

#### Use of PointConv:
Use the following to create pretrained model provided by authors.
> from learning3d.models import create_pointconv\
> PointConv = create_pointconv(classifier=True, pretrained='path of checkpoint')\
> ptconv = PointConv(emb_dims=1024, input_shape='bnc', input_channel_dim=6, classifier=True)

**OR**\
Use the following to create your own PointConv model.

> PointConv = create_pointconv(classifier=False, pretrained=None)\
> ptconv = PointConv(emb_dims=1024, input_shape='bnc', input_channel_dim=3, classifier=True)

PointConv variable is a class. Users can use it to create a sub-class to override *create_classifier* and *create_structure* methods in order to change PointConv's network architecture.

| Sr. No. | Variable | Data type | Choices | Use |
|:---:|:---:|:---:|:---:|:---:|
| 1. | emb_dims | Integer | 1024, 512 | Size of feature vector for each point | 
| 2. | input_shape | String | 'bnc' / 'bcn' | Shape of input point cloud |
| 3. | input_channel_dim | Integer | 3/6 | Define if point cloud contains only xyz co-ordinates or normals and colors as well |
| 4. | classifier | Boolean | True / False | Choose if you want to use a classifier with PointConv |
| 5. | pretrained | Boolean | String | Give path of the pretrained classifier model (only use it for weights given by authors) |

#### Use of Flow Estimation Network:
> from learning3d.models import FlowNet3D\
> flownet = FlowNet3D()

#### Use of Data Loaders:
> from learning3d.data_utils import ModelNet40Data, ClassificationData, RegistrationData, FlowData\
> modelnet40 = ModelNet40Data(train=True, num_points=1024, download=True)\
> classification_data = ClassificationData(data_class=ModelNet40Data())\
> registration_data = RegistrationData(algorithm='PointNetLK', data_class=ModelNet40Data(), partial_source=False, partial_template=False, noise=False)\
> flow_data = FlowData()

| Sr. No. | Variable | Data type | Choices | Use |
|:---:|:---:|:---:|:---:|:---:|
| 1. | train | Boolean | True / False | Split data as train/test set |
| 2. | num_points | Integer | 1024 | Number of points in each point cloud |
| 3. | download | Boolean | True / False | If data not available then download it |
| 4. | data_class | Object | - | Specify which dataset to use |
| 5. | algorithm | String | 'PointNetLK', 'PCRNet', 'DCP', 'iPCRNet' | Algorithm used for registration |
| 6. | partial_source | Boolean | True / False | Create partial source point cloud |
| 7. | partial_template | Boolean | True / False | Create partial template point cloud |
| 8. | noise | Boolean | True / False | Add noise in source point cloud |

#### Use Your Own Data:
> from learning3d.data_utils import UserData\
> dataset = UserData(application, data_dict)

|Sr. No. | Application | Required Key | Respective Value |
|:---:|:---:|:---:|:---:|
| 1. | 'classification' | 'pcs' | Point Clouds (BxNx3) |
|    |                  | 'labels' | Ground Truth Class Labels (BxN) |
| 2. | 'registration' | 'template' | Template Point Clouds (BxNx3) |
|    |                | 'source' | Source Point Clouds (BxNx3) |
|    |                | 'transformation' | Ground Truth Transformation (Bx4x4)|
| 3. | 'flow_estimation' | 'frame1' | Point Clouds (BxNx3) |
|    |                   | 'frame2' | Point Clouds (BxNx3) |
|    |                   | 'flow' | Ground Truth Flow Vector (BxNx3)|

#### Use of Loss Functions:
> from learning3d.losses import RMSEFeaturesLoss, FrobeniusNormLoss, ClassificationLoss, EMDLoss, ChamferDistanceLoss, CorrespondenceLoss\
> rmse = RMSEFeaturesLoss()\
> fn_loss = FrobeniusNormLoss()\
> classification_loss = ClassificationLoss()\
> emd = EMDLoss()\
> cd = ChamferDistanceLoss()\
> corr = CorrespondenceLoss()

| Sr. No. | Loss Type | Use |
|:---:|:---:|:---:|
| 1. | RMSEFeaturesLoss | Used to find root mean square value between two global feature vectors of point clouds |
| 2. | FrobeniusNormLoss | Used to find frobenius norm between two transfromation matrices |
| 3. | ClassificationLoss | Used to calculate cross-entropy loss | 
| 4. | EMDLoss | Earth Mover's distance between two given point clouds |
| 5. | ChamferDistanceLoss | Chamfer's distance between two given point clouds |
| 6. | CorrespondenceLoss | Computes cross entropy loss using the predicted correspondence and ground truth correspondence for each source point |

### To run codes from examples:
1. Copy the file from "examples" folder outside of the directory "learning3d"
2. Now, run the file. (ex. python test_pointnet.py)
- Your Directory/Location
	- learning3d
	- test_pointnet.py

### References:
1. [PointNet:](https://arxiv.org/abs/1612.00593) Deep Learning on Point Sets for 3D Classification and Segmentation
2. [Dynamic Graph CNN](https://arxiv.org/abs/1801.07829) for Learning on Point Clouds
3. [PPFNet:](https://arxiv.org/pdf/1802.02669.pdf) Global Context Aware Local Features for Robust 3D Point Matching
4. [PointConv:](https://arxiv.org/abs/1811.07246) Deep Convolutional Networks on 3D Point Clouds
5. [PointNetLK:](https://arxiv.org/abs/1903.05711) Robust & Efficient Point Cloud Registration using PointNet
6. [PCRNet:](https://arxiv.org/abs/1908.07906) Point Cloud Registration Network using PointNet Encoding
7. [Deep Closest Point:](https://arxiv.org/abs/1905.03304) Learning Representations for Point Cloud Registration
8. [PRNet:](https://arxiv.org/abs/1910.12240) Self-Supervised Learning for Partial-to-Partial Registration
9. [FlowNet3D:](https://arxiv.org/abs/1806.01411) Learning Scene Flow in 3D Point Clouds
10. [PCN:](https://arxiv.org/pdf/1808.00671.pdf) Point Completion Network
11. [RPM-Net:](https://arxiv.org/pdf/2003.13479.pdf) Robust Point Matching using Learned Features
12. [3D ShapeNets:](https://people.csail.mit.edu/khosla/papers/cvpr2015_wu.pdf) A Deep Representation for Volumetric Shapes
13. [DeepGMR:](https://arxiv.org/abs/2008.09088) Learning Latent Gaussian Mixture Models for Registration
14. [CMU:](https://arxiv.org/pdf/2010.16085.pdf) Correspondence Matrices are Underrated
15. [MaskNet:](https://arxiv.org/pdf/2010.09185.pdf) A Fully-Convolutional Network to Estimate Inlier Points
16. [MaskNet++:](https://www.sciencedirect.com/science/article/abs/pii/S0097849322000085) Inlier/outlier identification for two point clouds