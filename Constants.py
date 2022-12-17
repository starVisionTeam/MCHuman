from __future__ import division, absolute_import, print_function
import numpy as np
import os
body25_to_joint = np.array([11, 10, 9, 12, 13, 14, 4, 3, 2, 5, 6, 7, -1,
                            -1, -1, -1, -1, -1, -1, 0, 16, 15, 18, 17], dtype=np.int32)
cam_f = 5000
img_res = 512
cam_c = img_res/2,img_res/2
cam_R = np.eye(3, dtype=np.float32) * np.array([[1, -1, -1]], dtype=np.float32)
cam_tz = 10.0
cam_t = np.array([[0, 0, cam_tz]], dtype=np.float32)

vol_res = (128,192,128)
semantic_encoding_sigma = 0.05
smooth_kernel_size = 7
H_NORMALIZE=1
cmr_num_layers = 5
cmr_num_channels = 256

training_list_easy_fname = 'training_models_easy.txt'
training_list_hard_fname = 'training_models_hard.txt'

dataset_image_subfolder = 'image_data'
dataset_mesh_subfolder = 'mesh_data'
class Constants:
    def __init__(self,meshNormMargin=0.15 ):
        self.dim_w = 128
        self.dim_h = 192
        self.hb_ratio = self.dim_h/self.dim_w
        self.real_h = 1.0
        self.real_w = self.real_h /self.dim_h * self.dim_w
        self.voxel_size = self.real_h/self.dim_h # 1./192.
        self.tau = 0.5
        self.K = 100
        self.fill = True
        self.cmr_num_layers = 5
        self.cmr_num_channels = 256
        # for opendr rendering
        self.constBackground = 4294967295
        self.semantic_encoding_sigma = 0.05
        self.smooth_kernel_size = 7
        # for mesh voxelization
        self.h_normalize_half = self.real_h / 2.
        self.meshNormMargin = meshNormMargin # margin when normalizing mesh into H [-0.5,0.5]*(1-margin), W/D [-0.333,0.333]*(1-margin)
        self.threshH = self.h_normalize_half * (1-self.meshNormMargin)
        self.threshWD = self.h_normalize_half * self.dim_w/self.dim_h * (1-self.meshNormMargin)


        # black list images, due to wrong rendering with confusing background images, such as oversized human-heads
        self.black_list_images = ["092908", "092909", "092910", "092911",
                                  "090844", "090845", "090846", "090847"
                                 ]

        # tools
        self.smpl_vertex_code,self.smpl_tetras=self.read_smpl_constants()


    def read_smpl_constants(self):
        """Load smpl vertex code"""
        smpl_vtx_std = np.loadtxt(os.path.join('./data/vertices.txt'))
        min_x = np.min(smpl_vtx_std[:, 0])
        max_x = np.max(smpl_vtx_std[:, 0])
        min_y = np.min(smpl_vtx_std[:, 1])
        max_y = np.max(smpl_vtx_std[:, 1])
        min_z = np.min(smpl_vtx_std[:, 2])
        max_z = np.max(smpl_vtx_std[:, 2])

        smpl_vtx_std[:, 0] = (smpl_vtx_std[:, 0] - min_x) / (max_x - min_x)
        smpl_vtx_std[:, 1] = (smpl_vtx_std[:, 1] - min_y) / (max_y - min_y)
        smpl_vtx_std[:, 2] = (smpl_vtx_std[:, 2] - min_z) / (max_z - min_z)
        smpl_vertex_code = np.float32(np.copy(smpl_vtx_std))

        # """Load smpl faces & tetrahedrons"""
        # smpl_faces = np.loadtxt(os.path.join(folder, 'faces.txt'), dtype=np.int32) - 1
        # smpl_face_code = (smpl_vertex_code[smpl_faces[:, 0]] +
        #                   smpl_vertex_code[smpl_faces[:, 1]] + smpl_vertex_code[smpl_faces[:, 2]]) / 3.0
        smpl_tetras = np.loadtxt(os.path.join('./data', 'tetrahedrons.txt'), dtype=np.int32) - 1
        return smpl_vertex_code,smpl_tetras#, smpl_face_code, smpl_faces, smpl_tetras

consts = Constants()