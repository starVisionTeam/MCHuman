from torch.utils.data import Dataset
import numpy as np
import os
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import cv2
import torch
from PIL.ImageFilter import GaussianBlur
import trimesh
import logging
import scipy.io as sci
import pdb # pdb.set_trace()
import glob
import sys
import json
import copy
from Constants import consts
from model.util.io_util import save_samples_truncted_prob,save_volume
from util.obj_io import load_obj_data
def load_DeepHuman(meshPathsList):
    """
    XYZ direction
        X-right, Y-down, Z-inwards. All meshes face inwards along +Z.
    return
        a dict of ALL meshes, indexed by mesh-names
    """
    meshs = {}
    count = 0
    for meshPath in meshPathsList:
        print("Loading mesh %d/%d..." % (count, len(meshPathsList)))
        count += 1
        meshs[meshPath] = trimesh.load(meshPath)

    print("Sizes of meshs dict: %.3f MB." % (sys.getsizeof(meshs)/1024.)); pdb.set_trace()

    return meshs
class TrainDataset(Dataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser
    def __init__(self, opt, phase='train', allow_aug=True):
        self.opt = opt
        self.projection_mode = 'orthogonal'
        self.meshDirSearch=opt.meshDirSearch
        # Path setup
        self.B_MIN = np.array([-consts.real_w/2., -consts.real_h/2., -consts.real_w/2.])
        self.B_MAX = np.array([ consts.real_w/2.,  consts.real_h/2.,  consts.real_w/2.])
        self.is_train = (phase == 'train')
        self.load_size = self.opt.loadSize
        self.allow_aug = allow_aug
        self.smpl_vertex_code=consts.smpl_vertex_code
        self.num_sample_inout = self.opt.num_sample_inout # inside/outside 3d-sampling-points for occupancy query, default: 5000
        self.subjects = self.get_subjects() # a list of mesh paths for training or test
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con, saturation=opt.aug_sat,
                                   hue=opt.aug_hue)
        ])
        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.training_inds, self.testing_inds = self.get_training_test_indices(args=self.opt, shuffle=False)
        if phase=='test':
            dataNum = len(self.testing_inds)
            splitLen = int(np.ceil(1. * dataNum / self.opt.splitNum))
            splitRange = [self.opt.splitIdx * splitLen, min((self.opt.splitIdx + 1) * splitLen, dataNum)]
            self.testing_inds=self.testing_inds[splitRange[0]:splitRange[-1]]
        self.epochIdx = 0

    def get_subjects(self):
        """
        Return
            a list of mesh paths for training or test
        """
        personClothesDirs = glob.glob("%s/results_*" % (self.meshDirSearch)) # 202
        assert(len(personClothesDirs) == 202)
        meshPaths = [] # 6795
        for dirEach in personClothesDirs:
            meshPathsTmp = glob.glob("%s/*" % (dirEach))
            meshPathsTmp = [p+"/mesh.obj" for p in meshPathsTmp]
            meshPaths.extend(meshPathsTmp)
        meshNum = len(meshPaths) # 6795
        assert(meshNum == 6795)
        meshPaths_train = meshPaths[:int(np.ceil(self.opt.trainingDataRatio * meshNum))]  # 5436 unique mesh paths
        meshPaths_test  = meshPaths[int(np.floor(self.opt.trainingDataRatio * meshNum)):] # 1359 unique mesh paths
        return meshPaths_train if self.is_train else meshPaths_test

    def get_training_test_indices(self, args, shuffle=False):
        # sanity check for args.totalNumFrame
        totalNumFrameTrue = len(glob.glob(args.datasetDir+"/config/*.json"))
        assert((args.totalNumFrame == totalNumFrameTrue) or (args.totalNumFrame == totalNumFrameTrue+len(consts.black_list_images)//4))
        max_idx = args.totalNumFrame # total data number: N*M'*4 = 6795*4*4 = 108720
        indices = np.asarray(range(max_idx))
        assert(len(indices)%4 == 0)
        testing_flag = (indices >= int(args.trainingDataRatio*max_idx/4)*4)
        testing_inds = indices[testing_flag] # 21744 testing indices: array of [86976, ..., 108719]
        testing_inds = testing_inds.tolist()
        if shuffle: np.random.shuffle(testing_inds)
        assert(len(testing_inds) % 4 == 0)
        training_inds = indices[np.logical_not(testing_flag)] # 86976 training indices: array of [0, ..., 86975]
        training_inds = training_inds.tolist()
        if shuffle: np.random.shuffle(training_inds)
        assert(len(training_inds) % 4 == 0)
        return training_inds, testing_inds

    def __len__(self):
        return len(self.training_inds) if self.is_train else len(self.testing_inds)
    def rotateY_by_view(self, view_id):
        """
        input
            view_id: 0-front, 1-right, 2-back, 3-left
        """
        rotAngByViews = [0, -90., -180., -270.]
        angle = np.radians(rotAngByViews[view_id])
        ry = np.array([ [ np.cos(angle), 0., np.sin(angle)],
                        [            0., 1.,            0.],
                        [-np.sin(angle), 0., np.cos(angle)] ]) # (3,3)
        ry = np.transpose(ry)
        return ry
    def imageAugmen(self):
        pass
    def get_render(self, args, index, view_id,):
        render={}
        mask_path = "%s/maskImage/%06d.jpg" % (args.datasetDir, index)
        if not os.path.exists(mask_path):
            print("Can not find %s!!!" % (mask_path))
            pdb.set_trace()
        mask_data = np.round((cv2.imread(mask_path)[:,:,0]).astype(np.float32)/255.) # (1536, 1024)
        mask_data_padded = np.zeros((max(mask_data.shape), max(mask_data.shape)), np.float32) # (1536, 1536)
        mask_data_padded[:,mask_data_padded.shape[0]//2-min(mask_data.shape)//2:mask_data_padded.shape[0]//2+min(mask_data.shape)//2] = mask_data
        mask_data_padded = cv2.resize(mask_data_padded, (self.opt.loadSize,self.opt.loadSize), interpolation=cv2.INTER_NEAREST)
        mask_data_padded = Image.fromarray(mask_data_padded)
        mask_data_padded = transforms.ToTensor()(mask_data_padded).float()

        if 'img' in self.opt.keys_input:
            image_path = '%s/rgbImage/%06d.jpg' % (args.datasetDir, index)
            if not os.path.exists(image_path):
                print("Can not find %s!!!" % (image_path))
                pdb.set_trace()
            image = cv2.imread(image_path)[:,:,::-1] # (1536, 1024, 3), np.uint8, {0,...,255}
            image_padded = np.zeros((max(image.shape), max(image.shape), 3), np.uint8) # (1536, 1536, 3)
            image_padded[:,image_padded.shape[0]//2-min(image.shape[:2])//2:image_padded.shape[0]//2+min(image.shape[:2])//2,:] = image # (1536, 1536, 3)
            # resize to (512, 512, 3), np.uint8
            image_padded = cv2.resize(image_padded, (self.opt.loadSize, self.opt.loadSize))
            image_padded = Image.fromarray(image_padded)
            image_padded=self.to_tensor(image_padded)
            image_padded = mask_data_padded.expand_as(image_padded) * image_padded
            render['img']=image_padded
        if 'vmap' in self.opt.keys_input:
            # set paths
            smplSem_path = '%s/smplSem/%06d.jpg' % (args.datasetDir, index)
            if not os.path.exists(smplSem_path):
                print("Can not find %s!!!" % (smplSem_path))
                pdb.set_trace()
            # read data BGR -> RGB, np.uint8
            smplSem = cv2.imread(smplSem_path)[:,:,::-1] # (1536, 1024, 3), np.uint8, {0,...,255}
            smplSem_padded = np.zeros((max(smplSem.shape), max(smplSem.shape), 3), np.uint8) # (1536, 1536, 3)
            smplSem_padded[:,smplSem_padded.shape[0]//2-min(smplSem.shape[:2])//2:smplSem_padded.shape[0]//2+min(smplSem.shape[:2])//2,:] = smplSem # (1536, 1536, 3)
            # resize to (512, 512, 3), np.uint8
            smplSem_padded = cv2.resize(smplSem_padded, (self.opt.loadSize, self.opt.loadSize))
            smplSem_padded = Image.fromarray(smplSem_padded)
            smplSem_padded=self.to_tensor(smplSem_padded)
            smplSem_padded = mask_data_padded.expand_as(smplSem_padded) * smplSem_padded
            render['vmap']=smplSem_padded
        if 'normal_F' in self.opt.keys_input:
            # set paths
            normal_path = '%s_normal_F/%06d.jpg' % (args.normalDir, index)
            if not os.path.exists(normal_path):
                print("Can not find %s!!!" % (normal_path))
                pdb.set_trace()
            # read data BGR -> RGB, np.uint8
            normal = cv2.imread(normal_path)[:,:,::-1] # (1536, 1024, 3), np.uint8, {0,...,255}
            normal_padded = np.zeros((max(normal.shape), max(normal.shape), 3), np.uint8) # (1536, 1536, 3)
            normal_padded[:,normal_padded.shape[0]//2-min(normal.shape[:2])//2:normal_padded.shape[0]//2+min(normal.shape[:2])//2,:] = normal # (1536, 1536, 3)
            # resize to (512, 512, 3), np.uint8
            normal_padded = cv2.resize(normal_padded, (self.opt.loadSize, self.opt.loadSize))
            normal_padded = Image.fromarray(normal_padded)
            normal_padded= self.to_tensor(normal_padded)
            normal_padded = mask_data_padded.expand_as(normal_padded) * normal_padded
            render['normal_F']=normal_padded
        if 'normal_B' in self.opt.keys_input:
            # set paths
            normal_path = '%s_normal_B/%06d.jpg' % (args.normalDir, index)
            if not os.path.exists(normal_path):
                print("Can not find %s!!!" % (normal_path))
                pdb.set_trace()
            # read data BGR -> RGB, np.uint8
            normal = cv2.imread(normal_path)[:,:,::-1] # (1536, 1024, 3), np.uint8, {0,...,255}
            normal_padded = np.zeros((max(normal.shape), max(normal.shape), 3), np.uint8) # (1536, 1536, 3)
            normal_padded[:,normal_padded.shape[0]//2-min(normal.shape[:2])//2:normal_padded.shape[0]//2+min(normal.shape[:2])//2,:] = normal # (1536, 1536, 3)
            # resize to (512, 512, 3), np.uint8
            normal_padded = cv2.resize(normal_padded, (self.opt.loadSize, self.opt.loadSize))
            normal_padded = Image.fromarray(normal_padded)
            normal_padded= self.to_tensor(normal_padded)
            normal_padded = mask_data_padded.expand_as(normal_padded) * normal_padded
            render['normal_B']=normal_padded
            if 'img' in self.opt.keys_input:
                render['img']=torch.cat([render['img'],render['normal_F'],render['normal_B']],dim=0)
            else:
                render['img'] = torch.cat([render['normal_F'], render['normal_B']], dim=0)
        # ----- load calib and extrinsic Part-0 -----
        if True:
            # intrinsic matrix: ortho. proj. cam. model
            trans_intrinsic = np.identity(4) # trans intrinsic
            scale_intrinsic = np.identity(4) # ortho. proj. focal length
            scale_intrinsic[0,0] =  1./consts.h_normalize_half # const ==  2.
            scale_intrinsic[1,1] =  1./consts.h_normalize_half # const ==  2.
            scale_intrinsic[2,2] = -1./consts.h_normalize_half # const == -2.
            # extrinsic: model to cam R|t
            extrinsic        = np.identity(4)
            # randomRot        = np.array(dataConfig["randomRot"], np.float32) # by random R
            viewRot          = self.rotateY_by_view(view_id=view_id) # by view direction R
            # extrinsic[:3,:3] = np.dot(viewRot, randomRot)
            extrinsic[:3,:3] = viewRot
            # obtain the final calib and save calib/extrinsic
            intrinsic = np.matmul(trans_intrinsic, scale_intrinsic)
            calib     = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()
            extrinsic = torch.Tensor(extrinsic).float()
            # calib_list.append(calib)         # save calib
            # extrinsic_list.append(extrinsic) # save extrinsic
            render['calib']=calib
            render['extrinsic']=extrinsic
        return render
    def voxelization_normalization(self,verts,useMean=True,useScaling=True):
        """
        normalize the mesh into H [-0.5,0.5]*(1-margin), W/D [-0.333,0.333]*(1-margin)
        """
        vertsVoxelNorm = copy.deepcopy(verts)
        vertsMean, scaleMin = None, None
        if useMean:
            vertsMean = np.mean(vertsVoxelNorm,axis=0,keepdims=True) # (1, 3)
            vertsVoxelNorm -= vertsMean
        xyzMin = np.min(vertsVoxelNorm, axis=0); assert(np.all(xyzMin < 0))
        xyzMax = np.max(vertsVoxelNorm, axis=0); assert(np.all(xyzMax > 0))
        if useScaling:
            scaleArr = np.array([consts.threshWD/abs(xyzMin[0]), consts.threshH/abs(xyzMin[1]),consts.threshWD/abs(xyzMin[2]), consts.threshWD/xyzMax[0], consts.threshH/xyzMax[1], consts.threshWD/xyzMax[2]])
            scaleMin = np.min(scaleArr)
            vertsVoxelNorm *= scaleMin
        return vertsVoxelNorm, vertsMean, scaleMin
    def select_sampling_method_iccv_offline(self,index):
        """
        Path examples of the offline sampled query points 
            occu_sigma3.5_pts5k/088046_ep000_inPts.npy            0.0573 MB, np.float64, (2500, 3)
            occu_sigma3.5_pts5k/088046_ep000_outPts.npy           0.0573 MB, np.float64, (2500, 3)
        """
        # get inside and outside points
        inside_points_path  = "%s/%s/%06d_ep%03d_inPts.npy"  % (self.opt.datasetDir, self.opt.sampleType,index,self.epochIdx)
        outside_points_path = "%s/%s/%06d_ep%03d_outPts.npy" % (self.opt.datasetDir, self.opt.sampleType,index,self.epochIdx)
        assert(os.path.exists(inside_points_path) and os.path.exists(outside_points_path))
        inside_points    = np.load(inside_points_path)  # (N_in , 3), np.float64 
        outside_points   = np.load(outside_points_path) # (N_out, 3), np.float64
        num_sample_inout = inside_points.shape[0] + outside_points.shape[0]
        assert(num_sample_inout == self.num_sample_inout)
        # get samples and labels: {1-inside, 0-outside}
        samples = np.concatenate([inside_points, outside_points], 0).T # (3, n_in + n_out)
        labels  = np.concatenate([np.ones((1, inside_points.shape[0])), np.zeros((1, outside_points.shape[0]))], 1) # (1, n_in + n_out)
        samples = torch.Tensor(samples).float() # convert np.array to torch.Tensor 
        labels  = torch.Tensor(labels).float()
        return {'samples': samples, # (3, n_in + n_out), float XYZ coords are inside the 3d-volume of [self.B_MIN, self.B_MAX]
                'labels' : labels   # (1, n_in + n_out), 1.0-inside, 0.0-outside
               }
    def select_sampling_method_iccv_online(self,dataConfig):
        # read mesh
        # mesh = trimesh.load(dataConfig["meshPath"])
        meshPathTmp = dataConfig["meshPath"]
        assert(os.path.exists(meshPathTmp))
        mesh = trimesh.load(meshPathTmp)
        # normalize into volumes of X~[+-0.333], Y~[+-0.5], Z~[+-0.333]
        randomRot           = np.array(dataConfig["randomRot"], np.float32) # by random R
        mesh.vertex_normals = np.dot(mesh.vertex_normals, np.transpose(randomRot))
        mesh.face_normals   = np.dot(mesh.face_normals  , np.transpose(randomRot))
        mesh.vertices, _, _ = self.voxelization_normalization(np.dot(mesh.vertices,np.transpose(randomRot)))
        # uniformly sample points on mesh surface
        surface_points, _ = trimesh.sample.sample_surface(mesh, 4 * self.num_sample_inout) # (N1,3)
        # add gausian noise to surface points
        sample_points = surface_points + np.random.normal(scale=1.*self.opt.sigma/consts.dim_h, size=surface_points.shape) # (N1, 3)
        # uniformly sample inside the 128x192x128 volume, surface-points : volume-points ~ 16:1
        # volume_dims = np.array([[1.*consts.dim_w/consts.dim_h, 1.0, 1.*consts.dim_w/consts.dim_h]])
        # random_points = (np.random.rand(self.num_sample_inout//4, 3) - 0.5) * volume_dims # (N2, 3)
        length = self.B_MAX - self.B_MIN
        random_points = np.random.rand(self.num_sample_inout//4, 3) * length + self.B_MIN # (N2, 3)
        sample_points = np.concatenate([sample_points, random_points], 0) # (N1+N2, 3)
        np.random.shuffle(sample_points) # (N1+N2, 3)

        # determine {1, 0} occupancy ground-truth
        inside = mesh.contains(sample_points)
        inside_points  = sample_points[inside]
        outside_points = sample_points[np.logical_not(inside)]

        # constrain (n_in + n_out) <= self.num_sample_inout
        nin = inside_points.shape[0]
        # inside_points  =  inside_points[:self.num_sample_inout//2] if nin > self.num_sample_inout//2 else inside_points
        # outside_points = outside_points[:self.num_sample_inout//2] if nin > self.num_sample_inout//2 else outside_points[:(self.num_sample_inout - nin)]
        if nin > self.num_sample_inout//2:
            inside_points  =  inside_points[:self.num_sample_inout//2]
            outside_points = outside_points[:self.num_sample_inout//2]
        else:
            inside_points  = inside_points
            outside_points = outside_points[:(self.num_sample_inout - nin)]

        # get samples and labels: {1-inside, 0-outside}
        samples = np.concatenate([inside_points, outside_points], 0).T # (3, n_in + n_out)
        labels  = np.concatenate([np.ones((1, inside_points.shape[0])), np.zeros((1, outside_points.shape[0]))], 1) # (1, n_in + n_out)
        samples = torch.Tensor(samples).float() # convert np.array to torch.Tensor 
        labels  = torch.Tensor(labels).float()
        del mesh

        return {'samples': samples, # (3, n_in + n_out), float XYZ coords are inside the 3d-volume of [self.B_MIN, self.B_MAX]
                'labels' : labels   # (1, n_in + n_out), 1.0-inside, 0.0-outside
               }

    def get_mesh_voxels(self,idx):
        # init.
        dict_to_return = {}
        # ----- mesh voxels -----
        if 'meshVoxels' in self.opt.keys_input:
            # mesh voxels path
            mesh_volume_path = "%s/meshVoxels/%06d.mat" % (self.opt.datasetDir, idx)
            # sanity check
            if not os.path.exists(mesh_volume_path):
                print("Can not find %s!!!" % (mesh_volume_path))
                pdb.set_trace()
            # load data
            mesh_volume=sci.loadmat(mesh_volume_path)['meshVoxels'] # dtype('bool'), WHD-(128, 192, 128)
            # WHD -> DHW (as required by tensorflow)
            mesh_volume = np.transpose(mesh_volume, (2, 1, 0))
            # convert dtype
            mesh_volume = mesh_volume.astype(np.float32)        # (128,192,128)
            mesh_volume = torch.from_numpy(mesh_volume)[None,:] # (1,128,192,128)
            # add to dict
            dict_to_return["meshVoxels"] = mesh_volume # (C-1, D-128, H-192, W-128), 1-inside, 0-outsie
        if 'smplSemVoxels' in self.opt.keys_input:
            # mesh voxels path
            mesh_volume_path = "%s/smplSemVoxels/%06d.mat" % (self.opt.datasetDir, idx)
            # sanity check
            if not os.path.exists(mesh_volume_path):
                print("Can not find %s!!!" % (mesh_volume_path))
                pdb.set_trace()
            # load data
            # mesh_volume = np.load(mesh_volume_path)  # dtype('bool'), WHD-(128, 192, 128)
            mesh_volume = sci.loadmat(mesh_volume_path)['smplSemVoxels']
            # WHD -> DHW (as required by tensorflow)
            mesh_volume = np.transpose(mesh_volume, (3,2, 1, 0))
            # convert dtype
            mesh_volume = mesh_volume.astype(np.float32)  # (128,192,128)
            mesh_volume = torch.from_numpy(mesh_volume)  # (1,128,192,128)
            # add to dict
            dict_to_return["smplSemVoxels"] = mesh_volume  # (C-1, D-128, H-192, W-128), 1-inside, 0-outsie
        return dict_to_return

    def get_deepVoxels(self,idx):

        # init.
        dict_to_return = {}

        # ----- deepVoxels -----

        # set path
        deepVoxels_path = "%s/%06d_deepVoxels.npy" % (self.opt.deepVoxelsDir, idx)
        if not os.path.exists(deepVoxels_path):
            print("DeepVoxels: can not find %s!!!" % (deepVoxels_path))
            pdb.set_trace()

        # load npy, (C=8,W=32,H=48,D=32), C-XYZ, np.float32, only positive values
        deepVoxels_data = np.load(deepVoxels_path)

        # (C=8,W=32,H=48,D=32) to (C=8,D=32,H=48,W=32)
        deepVoxels_data = np.transpose(deepVoxels_data, (0,3,2,1))
        dict_to_return["deepVoxels"] = torch.from_numpy(deepVoxels_data)

        return dict_to_return

    def get_item(self, index):
        """
        data structures wrt index
            [pitch-idx, yaw-idx, mesh-idx] of sizes (len(self.pitch_list), len(self.yaw_list), len(self.subject))
        """

        # init.
        visualCheck_0 = False
        visualCheck_1 = False

        # ----- determine {volume_id, view_front_id, view_right_id, view_back_id, view_left_id} -----

        if not self.is_train: index += len(self.training_inds)
        if '%06d'%index in consts.black_list_images:
            return None

        volume_id  = index // 4 * 4
        view_id    = index - volume_id

        # ----- load "name", "index", "mesh_path", "b_min", "b_max", "in_black_list" -----

        # read config and get gt mesh path
        config_path = "%s/config/%06d.json" % (self.opt.datasetDir, index)
        with open(config_path) as f: dataConfig = json.load(f)
        meshPath = dataConfig["meshPath"]
        meshName = meshPath.split("/")[-3] + "+" + meshPath.split("/")[-2]
        res = {"name"         : meshName,
               "index"        : index,
               "mesh_path"    : meshPath,
               # "b_min"        : self.B_MIN,
               # "b_max"        : self.B_MAX,
               # "in_black_list": ("%06d"%index) in consts.black_list_images
              }
        # ----- load "img", "calib", "extrinsic", "mask" -----
        """
        render_data
            'img'      : [num_views, C, H, W] RGB, 3x512x512 images, float -1. ~ 1., bg is all ZEROS not -1.
            'calib'    : [num_views, 4, 4] calibration matrix
            'extrinsic': [num_views, 4, 4] extrinsic matrix
            'mask'     : [num_views, 1, H, W] for 1x512x512 masks, float 1.0-inside, 0.0-outside
        """
        render_data = self.get_render(args=self.opt, index=index,view_id=view_id)
        res.update(render_data)
        # ----- load "samples", "labels" if needed -----
        if 'smplSemVoxels' in self.opt.keys_input:
            res.update(self.get_mesh_voxels(index))
        if 'sample_data' in self.opt.keys_input and self.is_train: # default: 5000
            """
            sample_data is a dict.
                "samples" : (3, n_in + n_out), float XYZ coords are inside the 3d-volume of [self.B_MIN, self.B_MAX]
                "labels"  : (1, n_in + n_out), float 1.0-inside, 0.0-outside
            """
            sample_data = self.select_sampling_method_iccv_offline(index) if (self.is_train and (not self.opt.online_sampling)) else self.select_sampling_method_iccv_online(dataConfig)
            res.update(sample_data)
            # check "calib" by projecting "samplings" onto "img"
            if visualCheck_0:
                print("visualCheck_0: see if 'samples' can be properly projected to the 'img' by 'calib'...")

                # image RGB
                img = np.uint8((np.transpose(res['img'].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0) # HWC, BGR, float 0 ~ 255, de-normalized by mean 0.5 and std 0.5
                cv2.imwrite("./sample_images/%06d_img.png"%(index), img)

                # mask
                # mask = np.uint8(res['mask'][0,0].numpy() * 255.0) # (512, 512), 255 inside, 0 outside
                # cv2.imwrite("./sample_images/%06d_mask.png"%(index), mask)
                normal = np.uint8((np.transpose(res['normal'].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0) # HWC, BGR, float 0 ~ 255, de-normalized by mean 0.5 and std 0.5
                cv2.imwrite("./sample_images/%06d_normal.png"%(index), normal)
                # orthographic projection
                rot   = res['calib'][:3, :3]  # R_norm(-1,1)Cam_model, assuming that ortho. proj.: cam_f==256, c_x==c_y==256, img.shape==(512,512,3)
                trans = res['calib'][:3, 3:4] # T_norm(-1,1)Cam_model
                pts = torch.addmm(trans, rot, res['samples'][:, res['labels'][0] > 0.5])  # (3, 2500)
                pts = 0.5 * (pts.numpy().T + 1.0) * 512 # (2500,3), ortho. proj.: cam_f==256, c_x==c_y==256, img.shape==(512,512,3)
                imgProj = cv2.UMat(img)
                for p in pts: cv2.circle(imgProj, (p[0], p[1]), 2, (0,255,0), -1)
                cv2.imwrite("./sample_images/%06d_img_ptsProj.png"%(index), imgProj.get())     

                # save points in 3d
                samples_roted = torch.addmm(trans, rot, res['samples']) # (3, N)
                samples_roted[2,:] *= -1
                save_samples_truncted_prob("./sample_images/%06d_samples.ply"%(index), samples_roted.T, res["labels"].T)
        if 'smpl' in self.opt.keys_input:
            res.update(self.get_GT_smpl(view_id,dataConfig["meshPath"].replace('mesh.obj','smpl.obj'),dataConfig["meshNormMean"],dataConfig["randomRot"],dataConfig["meshNormScale"]))
            if self.opt.pre_smpl:
                res.update(self.get_pre_smpl(index))
        if "deepVoxel" in self.opt.keys_input:
            res.update(self.get_deepVoxels(index))

        return res
    def __getitem__(self, index):
        return self.get_item(index)
    def get_GT_smpl(self,view_id,smpl_path,meshNormMean,randomRot,meshNormScale):
        assert os.path.exists(smpl_path)
        gt_smpl=load_obj_data(smpl_path)['v']
        gt_smpl= copy.deepcopy(np.dot(gt_smpl-np.array(meshNormMean),np.array(randomRot).T)*np.array(meshNormScale))
        gt_smpl = self.inverseRotateY(points=gt_smpl, angle=-90*(view_id%4))
        return {'gt_smpl':torch.from_numpy(gt_smpl).float().T,'smpl_code':torch.from_numpy(self.smpl_vertex_code).float().T}
        # 'meshNormMean': torch.from_numpy(np.array(loadedConfig['meshNormMean'])).float(),
        # 'meshNormScale': torch.from_numpy(np.array(loadedConfig['meshNormScale'])).float(),
        # 'randomRotTrans': torch.from_numpy(np.array(loadedConfig['randomRot'])).T.float(),
    def inverseRotateY(self,points,angle):
        """
        Rotate the points by a specified angle., LEFT hand rotation
        """

        angle = np.radians(angle)
        ry = np.array([ [ np.cos(angle), 0., np.sin(angle)],
                        [            0., 1.,            0.],
                        [-np.sin(angle), 0., np.cos(angle)] ]) # (3,3)
        return np.dot(points, ry) # (N,3)
    def get_pre_smpl(self,index):
        '''
        pamir newPaMIR
        '''
        smpl_path="%s/%06d_meshRefined.obj"%(self.opt.pre_smpl_path,index)
        assert os.path.exists(smpl_path)
        pred_smpl=load_obj_data(smpl_path)['v']
        return {'pred_smpl':torch.from_numpy(pred_smpl).float().T,'smpl_code':torch.from_numpy(self.smpl_vertex_code).float().T}






