from __future__ import division, print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
import time
from torch.utils.cpp_extension import load

from scipy import ndimage
if os.path.exists('./cuda/cuda_mesh_voxelize.cu') and os.path.exists('./cuda/cuda_mesh_voxelize.cpp'):
    voxelize_cuda = load(
        name='voxelize_cuda',
        sources=['./cuda/cuda_mesh_voxelize.cpp', './cuda/cuda_mesh_voxelize.cu'],
        extra_ldflags=['-L/usr/local/cuda/targets/x86_64-linux/lib'],
        verbose=True)
elif os.path.exists('./cuda_voxelization/cuda/cuda_mesh_voxelize.cu') and \
        os.path.exists('./cuda_voxelization/cuda/cuda_mesh_voxelize.cpp'):
    voxelize_cuda = load(
        name='voxelize_cuda',
        sources=['./cuda_voxelization/cuda/cuda_mesh_voxelize.cpp',
                 './cuda_voxelization/cuda/cuda_mesh_voxelize.cu'],
        extra_ldflags=['-L/usr/local/cuda/targets/x86_64-linux/lib'],
        verbose=True)
else:
    raise IOError('Cannot find cuda/cuda_mesh_voxelize.cu and ./cuda/cuda_mesh_voxelize.cpp')


class MeshVoxelizationFunction(Function):
    """
    Definition of differentiable voxelization function
    Currently implemented only for cuda Tensors
    """
    @staticmethod
    def forward(ctx, vertices,vertices_faces,volume_res,H_NORMALIZE,smooth_kernel_size):
        """
        forward pass
        Output format: (batch_size, z_dims, y_dims, x_dims, channel_num)
        """
        ctx.volume_res = volume_res
        ctx.H_NORMALIZE = H_NORMALIZE
        ctx.smooth_kernel_size = smooth_kernel_size
        ctx.smpl_vertex_num = vertices.size()[0]
        ctx.device = vertices.device
        vertices_faces = vertices_faces.contiguous()
        occ_volume = torch.cuda.FloatTensor(ctx.volume_res[0], ctx.volume_res[1], ctx.volume_res[2]).fill_(0.0)
        occ_volume = voxelize_cuda.forward_voxelization(
            vertices_faces,occ_volume, H_NORMALIZE)
        return occ_volume.permute((2, 1, 0))
class MeshVoxelization(nn.Module):
    """
    Wrapper around the autograd function VoxelizationFunction
    """

    def __init__(self,  H_NORMALIZE,volume_res, smooth_kernel_size):
        super(MeshVoxelization, self).__init__()

        self.volume_res = volume_res
        self.H_NORMALIZE = H_NORMALIZE
        self.smooth_kernel_size = smooth_kernel_size
    def forward(self, vertices,faces):
        """
        Generate  volumes from the mesh vertices and faces
        """
        self.check_input(vertices)
        self.check_input(faces)
        vertices_faces = self.vertices_to_faces(vertices,faces)
        # smpl_face_center = self.calc_face_centers(smpl_faces)
        # smpl_face_normal = self.calc_face_normals(smpl_faces)

        occ_volume = MeshVoxelizationFunction.apply(vertices,vertices_faces,self.volume_res,self.H_NORMALIZE, self.smooth_kernel_size)

        return occ_volume  # (bzyxc --> bcdhw)

    def vertices_to_faces(self, vertices,faces):
        # assert (vertices.ndimension() == 3)
        vertices_ = vertices.reshape((-1, 3))
        return vertices_[faces.long()]

    def calc_face_centers(self, face_verts):
        assert len(face_verts.shape) == 4
        assert face_verts.shape[2] == 3
        assert face_verts.shape[3] == 3
        bs, nf = face_verts.shape[:2]
        face_centers = (face_verts[:, :, 0, :] + face_verts[:, :, 1, :] + face_verts[:, :, 2, :]) / 3.0
        face_centers = face_centers.reshape((bs, nf, 3))
        return face_centers

    def calc_face_normals(self, face_verts):
        assert len(face_verts.shape) == 4
        assert face_verts.shape[2] == 3
        assert face_verts.shape[3] == 3
        bs, nf = face_verts.shape[:2]
        face_verts = face_verts.reshape((bs * nf, 3, 3))
        v10 = face_verts[:, 0] - face_verts[:, 1]
        v12 = face_verts[:, 2] - face_verts[:, 1]
        normals = F.normalize(torch.cross(v10, v12), eps=1e-5)
        normals = normals.reshape((bs, nf, 3))
        return normals

    def check_input(self, x):
        if x.device == 'cpu':
            raise TypeError('Voxelization module supports only cuda tensors')
        if x.type() != 'torch.cuda.FloatTensor':
            raise TypeError('Voxelization module supports only float32  ')
def binary_fill_from_corner_3D(input, structure=None, output=None, origin=0):

    # now True means outside, False means inside
    mask = np.logical_not(input)

    # mark 8 corners as True
    tmp = np.zeros(mask.shape, bool)
    for xi in [0, tmp.shape[0]-1]:
        for yi in [0, tmp.shape[1]-1]:
            for zi in [0, tmp.shape[2]-1]:
                tmp[xi, yi, zi] = True

    # find connected regions from the 8 corners, to remove empty holes inside the voxels
    inplace = isinstance(output, np.ndarray)
    if inplace:
        ndimage.binary_dilation(tmp, structure=structure, iterations=-1,
                                mask=mask, output=output, border_value=0,
                                origin=origin)
        np.logical_not(output, output)
    else:
        output = ndimage.binary_dilation(tmp,structure=structure,iterations=-1,mask=mask,border_value=0,origin=origin)
        np.logical_not(output, output) # now 1 means inside, 0 means outside

        return output
class SematicVoxelizationFunction(Function):
    """
    Definition of differentiable voxelization function
    Currently implemented only for cuda Tensors
    """

    @staticmethod
    def forward(
            ctx, smpl_vertices,smpl_vertex_code, smpl_faces,occ_volume,
            volume_res,H_NORMALIZE, sigma, smooth_kernel_size):
        """
        forward pass
        Output format: (batch_size, z_dims, y_dims, x_dims, channel_num)
        """
        assert (smpl_vertices.size()[0] == smpl_vertex_code.size()[0])
        # ctx.batch_size = smpl_vertices.size()[0]
        ctx.volume_res = volume_res
        ctx.sigma = sigma
        ctx.H_NORMALIZE = H_NORMALIZE
        ctx.smooth_kernel_size = smooth_kernel_size
        ctx.smpl_vertex_num = smpl_vertices.size()[0]
        ctx.device = smpl_vertices.device

        smpl_vertices = smpl_vertices.contiguous()
        smpl_vertex_code = smpl_vertex_code.contiguous()
        smpl_faces = smpl_faces.contiguous()

        # occ_volume = torch.cuda.FloatTensor(ctx.batch_size, ctx.volume_res[0], ctx.volume_res[1], ctx.volume_res[2]).fill_(0.0)
        semantic_volume = torch.cuda.FloatTensor(ctx.volume_res[0], ctx.volume_res[1], ctx.volume_res[2],
                                                 3).fill_(0.0)
        weight_sum_volume = torch.cuda.FloatTensor(ctx.volume_res[0], ctx.volume_res[1], ctx.volume_res[2]).fill_(1e-3)
        # time_star= time.time()
        semantic_volume, weight_sum_volume = voxelize_cuda.forward_semantic_voxelization(
            smpl_vertices, smpl_vertex_code, smpl_faces,
            occ_volume, semantic_volume, weight_sum_volume, sigma,H_NORMALIZE)
        # time_end = time.time()
        # print('voxelize_cuda.forward_voxelization totally cost', time_end - time_star)

        return semantic_volume, weight_sum_volume


class SematicVoxelization(nn.Module):
    """
    Wrapper around the autograd function VoxelizationFunction
    """

    def __init__(self, smpl_vertex_code,   smpl_face_indices,H_NORMALIZE,volume_res, sigma, smooth_kernel_size):
        super(SematicVoxelization, self).__init__()
        assert (len(smpl_face_indices.shape) == 2)
        assert (smpl_face_indices.shape[1] == 3)

        self.volume_res = volume_res
        self.H_NORMALIZE = H_NORMALIZE
        self.sigma = sigma
        self.smooth_kernel_size = smooth_kernel_size
        # self.batch_size = batch_size
        # smpl_vertex_code_batch = np.tile(smpl_vertex_code, (batch_size, 1, 1))
        # smpl_face_indices_batch = np.tile(smpl_face_indices, (batch_size, 1, 1))
        smpl_vertex_code = torch.from_numpy(smpl_vertex_code).contiguous()
        smpl_face_indices = torch.from_numpy(smpl_face_indices).contiguous()
        self.register_buffer('smpl_vertex_code', smpl_vertex_code)
        self.register_buffer('smpl_face_indices', smpl_face_indices)

    def forward(self, smpl_vertices,occ_volume):
        """
        Generate semantic volumes from SMPL vertices
        """
        # assert (smpl_vertices.size()[0] == self.batch_size)
        self.check_input(smpl_vertices)
        smpl_faces = self.vertices_to_faces(smpl_vertices)
        semantic_volume,weight_sum_volume = SematicVoxelizationFunction.apply(
            smpl_vertices, self.smpl_vertex_code,smpl_faces,occ_volume,
            self.volume_res,self.H_NORMALIZE, self.sigma, self.smooth_kernel_size)
        return semantic_volume ,weight_sum_volume # (bzyxc --> bcdhw).permute((3, 0, 1, 2))

    def vertices_to_faces(self, vertices):
        # assert (vertices.ndimension() == 3)
        vertices_ = vertices.reshape((-1, 3))
        return vertices_[self.smpl_face_indices.long()]

    def vertices_to_tetrahedrons(self, vertices):
        assert (vertices.ndimension() == 3)
        bs, nv = vertices.shape[:2]
        device = vertices.device
        tets = self.smpl_tetraderon_indices_batch + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
        vertices_ = vertices.reshape((bs * nv, 3))
        return vertices_[tets.long()]

    def calc_face_centers(self, face_verts):
        assert len(face_verts.shape) == 4
        assert face_verts.shape[2] == 3
        assert face_verts.shape[3] == 3
        bs, nf = face_verts.shape[:2]
        face_centers = (face_verts[:, :, 0, :] + face_verts[:, :, 1, :] + face_verts[:, :, 2, :]) / 3.0
        face_centers = face_centers.reshape((bs, nf, 3))
        return face_centers

    def calc_face_normals(self, face_verts):
        assert len(face_verts.shape) == 4
        assert face_verts.shape[2] == 3
        assert face_verts.shape[3] == 3
        bs, nf = face_verts.shape[:2]
        face_verts = face_verts.reshape((bs * nf, 3, 3))
        v10 = face_verts[:, 0] - face_verts[:, 1]
        v12 = face_verts[:, 2] - face_verts[:, 1]
        normals = F.normalize(torch.cross(v10, v12), eps=1e-5)
        normals = normals.reshape((bs, nf, 3))
        return normals

    def check_input(self, x):
        if x.device == 'cpu':
            raise TypeError('Voxelization module supports only cuda tensors')
        if x.type() != 'torch.cuda.FloatTensor':
            raise TypeError('Voxelization module supports only float32 tensors')