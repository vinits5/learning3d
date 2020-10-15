import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from . import quaternion  # works with (w, x, y, z) quaternions
from scipy.spatial.transform import Rotation
from . import se3


def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat

def transform_point_cloud(point_cloud: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor):
    if len(rotation.size()) == 2:
        rot_mat = quat2mat(rotation)
    else:
        rot_mat = rotation
    return (torch.matmul(rot_mat, point_cloud.permute(0, 2, 1)) + translation.unsqueeze(2)).permute(0, 2, 1)

def convert2transformation(rotation_matrix: torch.Tensor, translation_vector: torch.Tensor):
    one_ = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(rotation_matrix.shape[0], 1, 1).to(rotation_matrix)    # (Bx1x4)
    transformation_matrix = torch.cat([rotation_matrix, translation_vector.unsqueeze(-1)], dim=2)                        # (Bx3x4)
    transformation_matrix = torch.cat([transformation_matrix, one_], dim=1)                                     # (Bx4x4)
    return transformation_matrix

def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)

def qmul_np(q, r):
    q = torch.from_numpy(q).contiguous()
    r = torch.from_numpy(r).contiguous()
    return qmul(q, r).numpy()

def euler_to_quaternion(e, order):
    """
    Convert Euler angles to quaternions.
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4

    e = e.reshape(-1, 3)

    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]

    rx = np.stack(
        (np.cos(x / 2), np.sin(x / 2), np.zeros_like(x), np.zeros_like(x)), axis=1
    )
    ry = np.stack(
        (np.cos(y / 2), np.zeros_like(y), np.sin(y / 2), np.zeros_like(y)), axis=1
    )
    rz = np.stack(
        (np.cos(z / 2), np.zeros_like(z), np.zeros_like(z), np.sin(z / 2)), axis=1
    )

    result = None
    for coord in order:
        if coord == "x":
            r = rx
        elif coord == "y":
            r = ry
        elif coord == "z":
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = qmul_np(result, r)

    # Reverse antipodal representation to have a non-negative "w"
    if order in ["xyz", "yzx", "zxy"]:
        result *= -1

    return result.reshape(original_shape)


class PNLKTransform:
    """ rigid motion """
    def __init__(self, mag=1, mag_randomly=False):
        self.mag = mag
        self.randomly = mag_randomly

        self.gt = None
        self.igt = None
        self.index = 0

    def generate_transform(self):
        # return: a twist-vector
        amp = self.mag
        if self.randomly:
            amp = torch.rand(1, 1) * self.mag
        x = torch.randn(1, 6)
        x = x / x.norm(p=2, dim=1, keepdim=True) * amp

        return x # [1, 6]

    def apply_transform(self, p0, x):
        # p0: [N, 3]
        # x: [1, 6]
        g = se3.exp(x).to(p0)   # [1, 4, 4]
        gt = se3.exp(-x).to(p0) # [1, 4, 4]

        p1 = se3.transform(g, p0)
        self.gt = gt.squeeze(0) #  gt: p1 -> p0
        self.igt = g.squeeze(0) # igt: p0 -> p1
        return p1

    def transform(self, tensor):
        x = self.generate_transform()
        return self.apply_transform(tensor, x)

    def __call__(self, tensor):
        return self.transform(tensor)


class RPMNetTransform:
    """ rigid motion """
    def __init__(self, mag=1, mag_randomly=False):
        self.mag = mag
        self.randomly = mag_randomly

        self.gt = None
        self.igt = None
        self.index = 0

    def generate_transform(self):
        # return: a twist-vector
        amp = self.mag
        if self.randomly:
            amp = torch.rand(1, 1) * self.mag
        x = torch.randn(1, 6)
        x = x / x.norm(p=2, dim=1, keepdim=True) * amp

        return x # [1, 6]

    def apply_transform(self, p0, x):
        # p0: [N, 3]
        # x: [1, 6]
        g = se3.exp(x).to(p0)   # [1, 4, 4]
        gt = se3.exp(-x).to(p0) # [1, 4, 4]

        p1 = se3.transform(g, p0[:, :3])

        if p0.shape[1] == 6:  # Need to rotate normals also
            g_n = g.clone()
            g_n[:, :3, 3] = 0.0
            n1 = se3.transform(g_n, p0[:, 3:6])
            p1 = torch.cat([p1, n1], axis=-1)

        self.gt = gt.squeeze(0) #  gt: p1 -> p0
        self.igt = g.squeeze(0) # igt: p0 -> p1
        return p1

    def transform(self, tensor):
        x = self.generate_transform()
        return self.apply_transform(tensor, x)

    def __call__(self, tensor):
        return self.transform(tensor)


class PCRNetTransform:
    def __init__(self, data_size, angle_range=45, translation_range=1):
        self.angle_range = angle_range
        self.translation_range = translation_range
        self.dtype = torch.float32
        self.transformations = [self.create_random_transform(torch.float32, self.angle_range, self.translation_range) for _ in range(data_size)]
        self.index = 0

    @staticmethod
    def deg_to_rad(deg):
        return np.pi / 180 * deg

    def create_random_transform(self, dtype, max_rotation_deg, max_translation):
        max_rotation = self.deg_to_rad(max_rotation_deg)
        rot = np.random.uniform(-max_rotation, max_rotation, [1, 3])
        trans = np.random.uniform(-max_translation, max_translation, [1, 3])
        quat = euler_to_quaternion(rot, "xyz")

        vec = np.concatenate([quat, trans], axis=1)
        vec = torch.tensor(vec, dtype=dtype)
        return vec

    @staticmethod
    def create_pose_7d(vector: torch.Tensor):
        # Normalize the quaternion.
        pre_normalized_quaternion = vector[:, 0:4]
        normalized_quaternion = F.normalize(pre_normalized_quaternion, dim=1)

        # B x 7 vector of 4 quaternions and 3 translation parameters
        translation = vector[:, 4:]
        vector = torch.cat([normalized_quaternion, translation], dim=1)
        return vector.view([-1, 7])

    @staticmethod
    def get_quaternion(pose_7d: torch.Tensor):
        return pose_7d[:, 0:4]

    @staticmethod
    def get_translation(pose_7d: torch.Tensor):
        return pose_7d[:, 4:]

    @staticmethod
    def quaternion_rotate(point_cloud: torch.Tensor, pose_7d: torch.Tensor):
        ndim = point_cloud.dim()
        if ndim == 2:
            N, _ = point_cloud.shape
            assert pose_7d.shape[0] == 1
            # repeat transformation vector for each point in shape
            quat = PCRNetTransform.get_quaternion(pose_7d).expand([N, -1])
            rotated_point_cloud = quaternion.qrot(quat, point_cloud)

        elif ndim == 3:
            B, N, _ = point_cloud.shape
            quat = PCRNetTransform.get_quaternion(pose_7d).unsqueeze(1).expand([-1, N, -1]).contiguous()
            rotated_point_cloud = quaternion.qrot(quat, point_cloud)

        return rotated_point_cloud

    @staticmethod
    def quaternion_transform(point_cloud: torch.Tensor, pose_7d: torch.Tensor):
        transformed_point_cloud = PCRNetTransform.quaternion_rotate(point_cloud, pose_7d) + PCRNetTransform.get_translation(pose_7d).view(-1, 1, 3).repeat(1, point_cloud.shape[1], 1)      # Ps' = R*Ps + t
        return transformed_point_cloud

    @staticmethod
    def convert2transformation(rotation_matrix: torch.Tensor, translation_vector: torch.Tensor):
        one_ = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(rotation_matrix.shape[0], 1, 1).to(rotation_matrix)    # (Bx1x4)
        transformation_matrix = torch.cat([rotation_matrix, translation_vector[:,0,:].unsqueeze(-1)], dim=2)                        # (Bx3x4)
        transformation_matrix = torch.cat([transformation_matrix, one_], dim=1)                                     # (Bx4x4)
        return transformation_matrix

    def __call__(self, template):
        self.igt = self.transformations[self.index]
        gt = self.create_pose_7d(self.igt)
        source = self.quaternion_rotate(template, gt) + self.get_translation(gt)
        return source


class DCPTransform:
    def __init__(self, angle_range=45, translation_range=1):
        self.angle_range = angle_range*(np.pi/180)
        self.translation_range = translation_range
        self.index = 0

    def generate_transform(self):
        self.anglex = np.random.uniform() * self.angle_range
        self.angley = np.random.uniform() * self.angle_range
        self.anglez = np.random.uniform() * self.angle_range
        self.translation = np.array([np.random.uniform(-self.translation_range, self.translation_range),
                                        np.random.uniform(-self.translation_range, self.translation_range),
                                        np.random.uniform(-self.translation_range, self.translation_range)])
        # cosx = np.cos(self.anglex)
        # cosy = np.cos(self.angley)
        # cosz = np.cos(self.anglez)
        # sinx = np.sin(self.anglex)
        # siny = np.sin(self.angley)
        # sinz = np.sin(self.anglez)
        # Rx = np.array([[1, 0, 0],
        #                 [0, cosx, -sinx],
        #                 [0, sinx, cosx]])
        # Ry = np.array([[cosy, 0, siny],
        #                 [0, 1, 0],
        #                 [-siny, 0, cosy]])
        # Rz = np.array([[cosz, -sinz, 0],
        #                 [sinz, cosz, 0],
        #                 [0, 0, 1]])
        # self.R_ab = Rx.dot(Ry).dot(Rz)
        # last_row = np.array([[0., 0., 0., 1.]])
        # self.igt = np.concatenate([self.R_ab, self.translation_ab.reshape(-1,1)], axis=1)
        # self.igt = np.concatenate([self.igt, last_row], axis=0)

    def apply_transformation(self, template):
        rotation = Rotation.from_euler('zyx', [self.anglez, self.angley, self.anglex])
        self.igt = rotation.apply(np.eye(3))
        self.igt = np.concatenate([self.igt, self.translation.reshape(-1,1)], axis=1)
        self.igt = torch.from_numpy(np.concatenate([self.igt, np.array([[0., 0., 0., 1.]])], axis=0)).float()
        source = rotation.apply(template) + np.expand_dims(self.translation, axis=0)
        return source

    def __call__(self, template):
        template = template.numpy()
        self.generate_transform()
        return torch.from_numpy(self.apply_transformation(template)).float()

class DeepGMRTransform:
    def __init__(self, angle_range=45, translation_range=1):
        self.angle_range = angle_range*(np.pi/180)
        self.translation_range = translation_range
        self.index = 0

    def generate_transform(self):
        self.anglex = np.random.uniform() * self.angle_range
        self.angley = np.random.uniform() * self.angle_range
        self.anglez = np.random.uniform() * self.angle_range
        self.translation = np.array([np.random.uniform(-self.translation_range, self.translation_range),
                                        np.random.uniform(-self.translation_range, self.translation_range),
                                        np.random.uniform(-self.translation_range, self.translation_range)])
        
    def apply_transformation(self, template):
        rotation = Rotation.from_euler('zyx', [self.anglez, self.angley, self.anglex])
        self.igt = rotation.apply(np.eye(3))
        self.igt = np.concatenate([self.igt, self.translation.reshape(-1,1)], axis=1)
        self.igt = torch.from_numpy(np.concatenate([self.igt, np.array([[0., 0., 0., 1.]])], axis=0)).float()
        source = rotation.apply(template) + np.expand_dims(self.translation, axis=0)
        return source

    def __call__(self, template):
        template = template.numpy()
        self.generate_transform()
        return torch.from_numpy(self.apply_transformation(template)).float()