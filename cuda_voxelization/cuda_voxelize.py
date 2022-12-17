from __future__ import division, print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
import time
from torch.utils.cpp_extension import load

if os.path.exists('./cuda/cuda_voxelize.cu') and os.path.exists('./cuda/cuda_voxelize.cpp'):
    sematic_voxelize_cuda = load(
        name='sematic_voxelize_cuda',
        sources=['./cuda/cuda_voxelize.cpp', './cuda/cuda_voxelize.cu'],
        extra_ldflags=['-L/usr/local/cuda/targets/x86_64-linux/lib'],
        verbose=True)
elif os.path.exists('./cuda_voxelization/cuda/cuda_voxelize.cu') and \
        os.path.exists('./cuda_voxelization/cuda/cuda_voxelize.cpp'):
    sematic_voxelize_cuda = load(
        name='sematic_voxelize_cuda',
        sources=['./cuda_voxelization/cuda/cuda_voxelize.cpp',
                 './cuda_voxelization/cuda/cuda_voxelize.cu'],
        extra_ldflags=['-L/usr/local/cuda/targets/x86_64-linux/lib'],
        verbose=True)
else:
    raise IOError('Cannot find cuda/cuda_voxelize.cu and ./cuda/cuda_voxelize.cpp')


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
        assert (smpl_vertices.size()[1] == smpl_vertex_code.size()[1])
        ctx.batch_size = smpl_vertices.size()[0]
        ctx.volume_res = volume_res
        ctx.sigma = sigma
        ctx.H_NORMALIZE = H_NORMALIZE
        ctx.smooth_kernel_size = smooth_kernel_size
        ctx.smpl_vertex_num = smpl_vertices.size()[1]
        ctx.device = smpl_vertices.device

        smpl_vertices = smpl_vertices.contiguous()
        smpl_vertex_code = smpl_vertex_code.contiguous()
        smpl_faces = smpl_faces.contiguous()

        # occ_volume = torch.cuda.FloatTensor(ctx.batch_size, ctx.volume_res[0], ctx.volume_res[1], ctx.volume_res[2]).fill_(0.0)
        semantic_volume = torch.cuda.FloatTensor(ctx.batch_size, ctx.volume_res[0], ctx.volume_res[1], ctx.volume_res[2],
                                                 3).fill_(0.0)
        weight_sum_volume = torch.cuda.FloatTensor(ctx.batch_size,ctx.volume_res[0], ctx.volume_res[1], ctx.volume_res[2]).fill_(1e-3)
        # time_star= time.time()
        occ_volume, semantic_volume, weight_sum_volume = sematic_voxelize_cuda.forward_semantic_voxelization(
            smpl_vertices, smpl_vertex_code, smpl_faces,
            occ_volume, semantic_volume, weight_sum_volume, sigma,H_NORMALIZE)
        # time_end = time.time()
        # print('voxelize_cuda.forward_voxelization totally cost', time_end - time_star)

        return semantic_volume


class SematicVoxelization(nn.Module):
    """
    Wrapper around the autograd function VoxelizationFunction
    """

    def __init__(self, smpl_vertex_code,   smpl_face_indices,H_NORMALIZE,volume_res, sigma, smooth_kernel_size, batch_size):
        super(SematicVoxelization, self).__init__()
        assert (len(smpl_face_indices.shape) == 2)
        assert (smpl_face_indices.shape[1] == 3)

        self.volume_res = volume_res
        self.H_NORMALIZE = H_NORMALIZE
        self.sigma = sigma
        self.smooth_kernel_size = smooth_kernel_size
        self.batch_size = batch_size
        smpl_vertex_code_batch = np.tile(smpl_vertex_code, (batch_size, 1, 1))
        smpl_face_indices_batch = np.tile(smpl_face_indices, (batch_size, 1, 1))
        smpl_vertex_code_batch = torch.from_numpy(smpl_vertex_code_batch).contiguous()
        smpl_face_indices_batch = torch.from_numpy(smpl_face_indices_batch).contiguous()
        self.register_buffer('smpl_vertex_code_batch', smpl_vertex_code_batch)
        self.register_buffer('smpl_face_indices_batch', smpl_face_indices_batch)

    def forward(self, smpl_vertices,occ_volume):
        """
        Generate semantic volumes from SMPL vertices
        """
        assert (smpl_vertices.size()[0] == self.batch_size)
        self.check_input(smpl_vertices)
        smpl_faces = self.vertices_to_faces(smpl_vertices)
        # smpl_face_center = self.calc_face_centers(smpl_faces)
        # smpl_face_normal = self.calc_face_normals(smpl_faces)
        smpl_surface_vertex_num = self.smpl_vertex_code_batch.size()[1]
        smpl_vertices_surface = smpl_vertices[:, :smpl_surface_vertex_num, :]
        semantic_volume,occ_volume = SematicVoxelizationFunction.apply(
            smpl_vertices_surface, self.smpl_vertex_code_batch,smpl_faces,occ_volume,
            self.volume_res,self.H_NORMALIZE, self.sigma, self.smooth_kernel_size)
        return semantic_volume.permute((0, 4, 1, 2, 3))  # (bzyxc --> bcdhw)

    def vertices_to_faces(self, vertices):
        assert (vertices.ndimension() == 3)
        bs, nv = vertices.shape[:2]
        device = vertices.device
        face = self.smpl_face_indices_batch + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
        vertices_ = vertices.reshape((bs * nv, 3))
        return vertices_[face.long()]

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