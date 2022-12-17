import sys
import os
import time
import json
import numpy as np
import cv2
from pymaf.models import SMPL, pymaf_net
from pymaf.core import path_config
from skimage import measure
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from model.util.sdf import *
from model.util.io_util import save_obj_mesh,save_obj_mesh_with_color
from util.obj_io import load_obj_data
import torchvision.transforms as transforms
from cuda_voxelization.cuda_mesh_voxelize import MeshVoxelization,binary_fill_from_corner_3D,SematicVoxelization
from model import ExplicitNet
from config.config import cfg
from model import ImplicitNet
import pdb # pdb.set_trace()
from Constants import  consts
from glob import glob
import pickle as pkl
from model.FBNet import define_G
from remove.imutils import process_image
import human_det
from neural_voxelization_layer.smpl_model import TetraSMPL
from neural_voxelization_layer.voxelize import Voxelization
from pytorch3d.structures import Meshes
from pytorch3d.renderer import ( look_at_view_transform, FoVOrthographicCameras, RasterizationSettings, MeshRenderer,
    MeshRasterizer, SoftSilhouetteShader)
class MCHumanDemo(object):
    def __init__(self):
        self.opt = cfg
        self.opt.merge_from_file('./config/MCHumanDemo.yaml')
        os.environ["CUDA_VISIBLE_DEVICES"] = self.opt.gpu_id
        self.cuda =torch.device('cuda')
        self.ENet = ExplicitNet(self.opt).to(device=self.cuda)
        print('Using Network: ', self.ENet.name)
        self.ENet=self.load_weight(self.opt.load_netV_checkpoint_path ,self.ENet)
        self.INet = ImplicitNet(self.opt).to(self.cuda)
        self.INet = self.load_weight(self.opt.load_netG_checkpoint_path ,self.INet)
        print('Using Network: ', self.INet.name)
        self.B_MIN = np.array([-consts.real_w/2., -consts.real_h/2., -consts.real_w/2.])
        self.B_MAX = np.array([ consts.real_w/2.,  consts.real_h/2.,  consts.real_w/2.])
        self.nmlFNet=define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance").to(self.cuda)
        self.nmlFNet=self.load_weight(self.opt.normal_netF_path ,self.nmlFNet)
        self.nmlBNet=define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance").to(self.cuda)
        self.nmlBNet=self.load_weight(self.opt.normal_netB_path ,self.nmlBNet)
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.opt.loadSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.det = human_det.Detection()
        self.hps = pymaf_net(path_config.SMPL_MEAN_PARAMS,
                        pretrained=True).to(self.cuda)
        self.hps.load_state_dict(torch.load(
            path_config.CHECKPOINT_FILE)['model'],
                            strict=True)
        self.hps.eval()
        self.smpl_model = SMPL('./data/smpl/',batch_size=1,create_transl=False).to(self.cuda)
        self.faces = torch.Tensor(
            self.smpl_model.faces.astype(np.int16)).long().unsqueeze(0).to(
            self.cuda)
        self.tet_smpl = TetraSMPL('./data/GCMR/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
                                  './data/GCMR/tetra_smpl.npz').to(self.cuda)
        self.voxelization = Voxelization(consts.smpl_vertex_code,consts.smpl_tetras ,
                                         sigma=0.05,
                                         smooth_kernel_size=7,
                                         batch_size=1).to(self.cuda)
        self.openpose_dir='/home/sunjc0306/openpose-master'
        R, T = look_at_view_transform(2, 0, 0)
        camera = FoVOrthographicCameras(device=self.cuda,R=R,T=T, scale_xyz=(1 * np.ones(3),))
        if True:
            raster_settings_silhouette = RasterizationSettings(
                image_size=512,
                blur_radius=np.log(1. / 1e-4 - 1.) * 5e-5,
                faces_per_pixel=50,
                cull_backfaces=True,
            )

            silhouetteRas = MeshRasterizer(
                cameras=camera, raster_settings=raster_settings_silhouette)
            self.renderer = MeshRenderer(rasterizer=silhouetteRas,
                                    shader=SoftSilhouetteShader())
    def load_weight(self,checkpoint_path,model):
        if checkpoint_path is not None:
            print('loading for net  ...', checkpoint_path)
            assert (os.path.exists(checkpoint_path))
            try:
                model.load_state_dict(
                self.load_from_multi_GPU(path=checkpoint_path))
            except :
                model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.cuda))
        return model
    def load_from_multi_GPU(self, path):
        # original saved file with DataParallel
        state_dict = torch.load(path)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        return new_state_dict
    def demo(self,img_path,results_path,save_inter=True):
        input_data={}
        img_icon, img_hps, img_ori, img_mask, uncrop_param = process_image(img_path,self.det, self.opt.loadSize)
        img_icon=img_icon.unsqueeze(0).to(self.cuda)
        img_mask=img_mask.unsqueeze(0).to(self.cuda)
        img_name = img_path.split("/")[-1].rsplit(".", 1)[0]
        if save_inter:
            img_BGR = ((np.transpose(img_icon[0].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5) * 255.).astype(np.uint8)[:, :,::-1]
            cv2.imwrite('%s/%s_rgb.jpg' % (results_path,img_name), img_BGR)
            # return 0.0
            img_BGR = ((np.transpose(img_mask[0].detach().cpu().numpy(), (1, 2, 0))) * 255.).astype(np.uint8)[:, :, ::-1]
            cv2.imwrite('%s/%s_mask.jpg' % (results_path,img_name), img_BGR)
        start_time=time.time()
        with torch.no_grad():
            normal_F=self.nmlFNet(img_icon)* img_mask
            normal_B=self.nmlBNet(img_icon)* img_mask
            preds_dict = self.hps(img_hps.to(self.cuda))
            output = preds_dict['smpl_out'][-1]
            scale, tranX, tranY = output['theta'][0, :3]
            data = {}
            data['betas'] = output['pred_shape']
            data['body_pose'] = output['rotmat'][:, 1:]
            data['global_orient'] = output['rotmat'][:, 0:1]
            data['smpl_verts'] = output['verts']
            trans = torch.tensor([tranX, tranY, 0.0]).to(self.cuda)
            data['scale'] = scale
            data['trans'] = trans
        input_data['img']=torch.cat([img_icon,normal_F,normal_B],dim=1)
        if save_inter:
            img_BGR = ((np.transpose(normal_F[0].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5) * 255.).astype(np.uint8)[:, :,::-1]
            cv2.imwrite('%s/%s_normal_F.jpg' % (results_path,img_name), img_BGR)
            img_BGR = ((np.transpose(normal_B[0].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5) * 255.).astype(np.uint8)[:, :,::-1]
            cv2.imwrite('%s/%s_normal_B.jpg' % (results_path,img_name), img_BGR)
            smpl_out = self.smpl_model(betas=data['betas'], body_pose=data['body_pose'], global_orient=data['global_orient'],pose2rot=False)
            smpl_verts = (smpl_out.vertices * data['scale']) + data['trans']
            smpl_verts *= torch.tensor([1.0, -1.0, -1.0]).to(self.cuda)
            save_obj_mesh('%s/%s_init_smpl.obj' % (results_path,img_name), smpl_verts[0].detach().cpu().numpy(),
                          self.faces[0].detach().cpu().numpy())
        img_BGR = ((np.transpose(img_icon[0].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5) * 255.).astype(np.uint8)[:, :,
                  ::-1]
        cv2.imwrite('%s/temp/temp_rgb.jpg' % (os.getcwd()), img_BGR)
        cmd = "cd {0}; ./build/examples/openpose/openpose.bin --image_dir {1} --write_json {2}".format(
                self.openpose_dir,
                os.getcwd() + '/temp',
                os.getcwd() + '/temp')
        os.system(cmd)
        keypoints_path = os.getcwd() + '/temp/' + 'temp_rgb_keypoints.json'
        with open(keypoints_path) as fp:
            keypointsdata = json.load(fp)
        keypoints = []
        if 'people' in keypointsdata:
            for idx, person_data in enumerate(keypointsdata['people']):
                kp_data = np.array(person_data['pose_keypoints_2d'], dtype=np.float32)
                kp_data = kp_data.reshape([-1, 3])
                kp_data[:, 0] = kp_data[:, 0] * 2 / self.opt.loadSize - 1.0
                kp_data[:, 1] = kp_data[:, 1] * 2 / self.opt.loadSize - 1.0
                keypoints.append(kp_data)
        if len(keypoints) == 0:
            keypoints.append(np.zeros([25, 3]))
        keypoints = torch.from_numpy(keypoints[0])[None, :, :].float().to(self.cuda)
        optimed_betas, optimed_pose, optimed_orient, optimed_trans, smpl_verts=\
            self.optm_smpl_param(img_mask[0].permute([1,2,0]),keypoints,data['betas'],data['body_pose'],data['global_orient'],data['scale'],data['trans'],iter_num=100)
        with torch.no_grad():
            # work_path = '%s/../ICON/%s_smplRefined.pkl' % (results_path, img_name)
            # if os.path.exists(work_path):
            #     print(f'load {work_path}')
            #     with open(work_path, "rb") as f:
            #         data_smplRefined = pkl.load(f)
            #         optimed_orient = torch.from_numpy(data_smplRefined['global_orient'])[None,None].to(self.cuda)
            #         optimed_pose = torch.from_numpy(data_smplRefined['body_pose'])[None].to(self.cuda)
            #         optimed_betas = torch.from_numpy(data_smplRefined['betas'])[None].to(self.cuda)
            #         optimed_trans = torch.from_numpy(data_smplRefined['trans']).to(self.cuda)
            #         data['scale'] = torch.from_numpy(data_smplRefined['scale']).to(self.cuda)
            gt_vert_cam = data['scale'] * self.tet_smpl(torch.cat([optimed_orient,optimed_pose],dim=1), optimed_betas) + optimed_trans
            # gt_vert_cam = data['scale'] * self.tet_smpl(torch.cat([data['global_orient'],data['body_pose']],dim=1), data['betas']) + data['trans']
            vol = self.voxelization(gt_vert_cam/2)
        if save_inter:
            save_obj_mesh('%s/%s_optim_smpl.obj' % (results_path,img_name), smpl_verts[0].detach().cpu().numpy(),
                          self.faces[0].detach().cpu().numpy())

        with torch.no_grad():
            self.ENet.filter(255*vol)
            input_data['deepVoxels'] = self.ENet.im_feat_list[-1]  # torch.float32, (B=1,C=8,D=32,H=48,W=32), etc. deepVoxels
        if save_inter:
            deepVoxels = input_data['deepVoxels'][0].detach().cpu().numpy()  # (C=8,D=32,H=48,W=32)
            deepVoxels = np.transpose(deepVoxels, (
                0, 3, 2, 1))  # (C=8,W=32,H=48,D=32), C-XYZ, np.float32, has both posi/neg values
            np.save('%s/%s_deepVoxels.npy' % (results_path,img_name), deepVoxels)  # 1.6M
        # get est. occu.
        if save_inter:
            with torch.no_grad():
                save_path ='%s/%s_meshCoarse.obj' % (results_path,img_name)
                self.ENet.est_occu()
                pred_occ = self.ENet.get_preds()  # torch.float32, BCDHW, (B,1,128,192,128), est. occupancy
                pred_occ = pred_occ[0, 0].detach().cpu().numpy()  # DHW
                pred_occ = np.transpose(pred_occ, (2, 1, 0))  # (W=128,H=192,D=128), XYZ, np.float32, 0. ~ 1. # WHD, XYZ
                verts, faces, normals, _ = measure.marching_cubes_lewiner(pred_occ,level=0.5)  # https://scikit-image.org/docs/dev/api/skimage.measure.html#marching-cubes-lewiner
                verts = verts * 2.0  # this is only to match the verts_canonization function
                # verts = verts_canonization(verts=verts, dim_w=pred_occ.shape[0], dim_h=pred_occ.shape[1])
                save_obj_mesh(save_path,verts,faces[:,::-1])
        if True:
            projection_matrix = np.identity(4)
            projection_matrix[0, 0] = 1. / consts.h_normalize_half  # const ==  2.
            projection_matrix[1, 1] = 1. / consts.h_normalize_half  # const ==  2.
            projection_matrix[2, 2] = -1. / consts.h_normalize_half  # const == -2., to get inverse depth
            calib = torch.Tensor(projection_matrix).float()
            input_data['calib'] = calib.unsqueeze(0).to(self.cuda)
        with torch.no_grad():
            self.INet.filter(input_data['img'])
            try:
                coords, mat = create_grid(self.opt.resolution_x, self.opt.resolution_y, self.opt.resolution_z,
                                          self.B_MIN, self.B_MAX, transform=None)

                def eval_func(points):
                    points = np.expand_dims(points, axis=0)  # (1,         3, num_samples)
                    points = np.repeat(points, 1, axis=0)  # (num_views, 3, num_samples)
                    samples = torch.from_numpy(points).to(device=self.cuda).float()  # (num_views, 3, num_samples)
                    self.INet.query(points=samples, calibs=input_data['calib'],
                                    deepVoxels=input_data['deepVoxels'])  # calib_tensor is (num_views, 4, 4)
                    pred = self.INet.preds[0][0]  # (num_samples,)
                    return pred.detach().cpu().numpy()

                sdf = eval_grid_octree(coords, eval_func, num_samples=self.opt.num_samples)
                verts, faces, normals, values = measure.marching_cubes_lewiner(sdf, 0.5)
                # transform verts into world coordinate system
                verts = np.matmul(mat[:3, :3], verts.T) + mat[:3,3:4]  # (3,N), convert verts from voxel-space into mesh-coords
                verts = verts.T  # (N,3)

                obj_result = '%s/%s_meshRefined.obj' % (results_path, img_name)
                save_obj_mesh(obj_result, verts*np.array([1.,-1.,-1.]), faces[:,::-1])

                verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=self.cuda).float()  # (1, N, 3)
                xyz_tensor = self.INet.projection(verts_tensor, input_data['calib'][:1])  # （1, 3, N） Tensor of xyz coordinates in the image plane of (-1,1) zone and of the-first-view
                uv = xyz_tensor[:, :2, :]  # (1, 2, N) for xy, float -1 ~ 1
                color = self.INet.index(img_icon[:1], uv).detach().cpu().numpy()[0].T  # (N, 3), RGB, float -1 ~ 1
                color = color * 0.5 + 0.5  # (N, 3), RGB, float 0 ~ 1
                save_obj_mesh_with_color('%s/%s_color.obj' % (results_path, img_name), verts, faces, color)
                # return sdf
            except Exception as e:
                print(e)
                print('Can not create marching cubes at this time.')
        return time.time()-start_time

    def optm_smpl_param(self,gt_silhouette, keypoint, betas, pose, global_orient, scale, trans, iter_num=0):
        # assert iter_num > 0
        optimed_pose = torch.tensor(pose, device=self.cuda, requires_grad=True)  # [1,23,3,3]
        optimed_trans = torch.tensor(trans, device=self.cuda, requires_grad=True)  # [3]
        optimed_betas = torch.tensor(betas, device=self.cuda, requires_grad=False)  # [1,10]
        optimed_orient = torch.tensor(global_orient, device=self.cuda, requires_grad=True)  # [1,1,3,3]

        optimizer_smpl = torch.optim.SGD([optimed_pose, optimed_trans, optimed_betas, optimed_orient], lr=3e-2,
                                         momentum=0.9)
        scheduler_smpl = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_smpl, mode='min', factor=0.5, verbose=0,
                                                                    min_lr=1e-5, patience=5)
        smpl_out = self.smpl_model(betas=optimed_betas, body_pose=optimed_pose, global_orient=optimed_orient,
                                   pose2rot=False)
        smpl_verts = (smpl_out.vertices * scale) + optimed_trans
        for i in range(iter_num):
            smpl_out = self.smpl_model(betas=optimed_betas, body_pose=optimed_pose, global_orient=optimed_orient,
                                  pose2rot=False)
            smpl_verts = (smpl_out.vertices * scale) + optimed_trans
            smpl_verts *= torch.tensor([1.0, -1.0, -1.0]).to(self.cuda)
            mesh = Meshes(smpl_verts, self.faces).to(self.cuda)
            silhouette =self.renderer(mesh)[0, :, :, 3:]
            openpose_joints = (smpl_out.joints[:, :25, :] * scale) +optimed_trans
            # if i % 10 == 0:
            #     cv2.imwrite("./debug/%dsilhouette.png" % (i),
            #                 np.uint8((silhouette > 0).detach().cpu().numpy()) * 255)
            loss_pose = torch.mean(
                (keypoint[:, :, 2:] * openpose_joints[:, :, :2] - keypoint[:, :, 2:] * keypoint[:, :, :2]) ** 2)
            diff_S = torch.abs(silhouette - gt_silhouette)
            loss_sil = diff_S.mean()
            loss =loss_pose* 30.0 + loss_sil * 1.0#+F.l1_loss(optimed_pose,pose)+F.l1_loss(optimed_betas,betas)
            optimizer_smpl.zero_grad()
            loss.backward()
            optimizer_smpl.step()
            scheduler_smpl.step(loss)
            # print('Iter No.%d: loss = %f, loss_sil = %f, loss_pose = %f' %
            #       (i, loss.item(), loss_sil.item(), loss_pose.item()))

        return optimed_betas, optimed_pose, optimed_orient, optimed_trans, smpl_verts

    def demo_w_b_s(self,image_path,mask_path,smplPath,normal_F_Path,normal_B_Path):
        # smpl_vpath="./optim/FRONT_smpl_v.mat"
        smplMesh = load_obj_data(smplPath)
        # save_obj_data(smplNew, smplPathNew)  # notice that face's vertex idx should start from +1, not 0
        voxeltool = MeshVoxelization(1., (128, 192, 128), 7)
        semantictool = SematicVoxelization(consts.smpl_vertex_code, smplMesh['f'], 1., (128, 192, 128), 0.05, 7).cuda()
        occ_volume = voxeltool(torch.from_numpy(smplMesh['v']).float().cuda(),
                               torch.from_numpy(smplMesh['f']).float().cuda())
        occ_volume = binary_fill_from_corner_3D(occ_volume.cpu().numpy().astype(np.uint8))
        smplSemVoxels, _ = semantictool(torch.from_numpy(smplMesh['v']).float().cuda(),
                                        torch.from_numpy(occ_volume).float().cuda())
        render={}
        render['smplSemVoxels']=smplSemVoxels
        if not os.path.exists(mask_path):
            print("Can not find %s!!!" % (mask_path))
            pdb.set_trace()
        mask_data = np.round((cv2.imread(mask_path)[:,:,0]).astype(np.float32)/255.) # (1536, 1024)
        mask_data_padded = np.zeros((max(mask_data.shape), max(mask_data.shape)), np.float32) # (1536, 1536)
        mask_data_padded[:,mask_data_padded.shape[0]//2-min(mask_data.shape)//2:mask_data_padded.shape[0]//2+min(mask_data.shape)//2] = mask_data
        mask_data_padded = cv2.resize(mask_data_padded, (self.opt.loadSize,self.opt.loadSize), interpolation=cv2.INTER_NEAREST)
        mask_data_padded = Image.fromarray(mask_data_padded)
        mask_data_padded = transforms.ToTensor()(mask_data_padded).float()

        if not os.path.exists(image_path):
            print("Can not find %s!!!" % (image_path))
            pdb.set_trace()
        image = cv2.imread(image_path)[:, :, ::-1]  # (1536, 1024, 3), np.uint8, {0,...,255}
        image_padded = np.zeros((max(image.shape), max(image.shape), 3), np.uint8)  # (1536, 1536, 3)
        image_padded[:,
        image_padded.shape[0] // 2 - min(image.shape[:2]) // 2:image_padded.shape[0] // 2 + min(image.shape[:2]) // 2,
        :] = image  # (1536, 1536, 3)
        # resize to (512, 512, 3), np.uint8
        image_padded = cv2.resize(image_padded, (self.opt.loadSize, self.opt.loadSize))
        image_padded = Image.fromarray(image_padded)
        image_padded = self.to_tensor(image_padded)
        image_padded = mask_data_padded.expand_as(image_padded) * image_padded
        # render['img'] = image_padded
        if not os.path.exists(normal_F_Path):
            print("Can not find %s!!!" % (normal_F_Path))
            pdb.set_trace()
        # read data BGR -> RGB, np.uint8
        normal = cv2.imread(normal_F_Path)[:, :, ::-1]  # (1536, 1024, 3), np.uint8, {0,...,255}
        normal_F = np.zeros((max(normal.shape), max(normal.shape), 3), np.uint8)  # (1536, 1536, 3)
        normal_F[:,
        normal_F.shape[0] // 2 - min(normal.shape[:2]) // 2:normal_F.shape[0] // 2 + min(normal.shape[:2]) // 2,
        :] = normal  # (1536, 1536, 3)
        # resize to (512, 512, 3), np.uint8
        normal_F = cv2.resize(normal_F, (self.opt.loadSize, self.opt.loadSize))
        normal_F = Image.fromarray(normal_F)
        normal_F = self.to_tensor(normal_F)
        normal_F = mask_data_padded.expand_as(normal_F) * normal_F

        if not os.path.exists(normal_B_Path):
            print("Can not find %s!!!" % (normal_B_Path))
            pdb.set_trace()
        # read data BGR -> RGB, np.uint8
        normal = cv2.imread(normal_B_Path)[:, :, ::-1]  # (1536, 1024, 3), np.uint8, {0,...,255}
        normal_B = np.zeros((max(normal.shape), max(normal.shape), 3), np.uint8)  # (1536, 1536, 3)
        normal_B[:,
        normal_B.shape[0] // 2 - min(normal.shape[:2]) // 2:normal_B.shape[0] // 2 + min(normal.shape[:2]) // 2,
        :] = normal  # (1536, 1536, 3)
        # resize to (512, 512, 3), np.uint8
        normal_B = cv2.resize(normal_B, (self.opt.loadSize, self.opt.loadSize))
        normal_B = Image.fromarray(normal_B)
        normal_B = self.to_tensor(normal_B)
        normal_B = mask_data_padded.expand_as(normal_B) * normal_B
        render['img'] = torch.cat([image_padded, normal_F, normal_B], dim=0)
        if True:
            projection_matrix = np.identity(4)
            projection_matrix[0, 0] = 1. / consts.h_normalize_half  # const ==  2.
            projection_matrix[1, 1] = 1. / consts.h_normalize_half  # const ==  2.
            projection_matrix[2, 2] = -1. / consts.h_normalize_half  # const == -2., to get inverse depth
            calib = torch.Tensor(projection_matrix).float()
            render['calib'] = calib
        return render
    def test_demo(self,data,obj_result):
        with torch.no_grad():
            self.ENet.eval()
            smplSemVoxels = data['smplSemVoxels'].to(device=self.cuda)
            smplSemVoxels = smplSemVoxels.permute((3, 2, 1, 0)).unsqueeze(0)  # (V=1, C=3, H=512, W=512)
            self.ENet.filter(smplSemVoxels)
            deepVoxels = self.ENet.im_feat_list[-1]
            self.INet.filter(data['img'])
            try:
                coords, mat = create_grid(self.opt.resolution_x, self.opt.resolution_y, self.opt.resolution_z,
                                          self.B_MIN, self.B_MAX, transform=None)
                def eval_func(points):
                    points = np.expand_dims(points, axis=0)  # (1,         3, num_samples)
                    points = np.repeat(points, 1, axis=0)  # (num_views, 3, num_samples)
                    samples = torch.from_numpy(points).to(device=self.cuda).float()  # (num_views, 3, num_samples)
                    self.INet.query(points=samples, calibs=data['calib'],
                                    deepVoxels=deepVoxels)  # calib_tensor is (num_views, 4, 4)
                    pred = self.INet.preds[0][0]  # (num_samples,)
                    return pred.detach().cpu().numpy()
                sdf = eval_grid_octree(coords, eval_func, num_samples=self.opt.num_samples)
                verts, faces, normals, values = measure.marching_cubes_lewiner(sdf, 0.5)
                verts = np.matmul(mat[:3, :3], verts.T) + mat[:3,3:4]  # (3,N), convert verts from voxel-space into mesh-coords
                verts = verts.T  # (N,3)
                save_obj_mesh(obj_result, verts, faces)
                # return sdf
            except Exception as e:
                print(e)
                print('Can not create marching cubes at this time.')
if __name__ == '__main__':
    test=MCHumanDemo()
    # rgbPath='/media/sunjc0306/KESU/dataset/Buff/BuffRender/rgbImage/000000.jpg'
    # maskPath='/media/sunjc0306/KESU/dataset/Buff/BuffRender/maskImage/000000.jpg'
    # smplPath="/media/sunjc0306/KESU/dataset/Buff/WoGtPaMIR/000000_smpl.obj"
    # normal_F_Path='/media/sunjc0306/KESU/dataset/Buff/normal_normal_F/000000.jpg'
    # normal_B_Path='/media/sunjc0306/KESU/dataset/Buff/normal_normal_B/000000.jpg'
    img_path_list=glob('/home/sunjc0306/HEI-Human/Pin/*')
    img_path_list.sort(reverse=True)
    count=0
    timeStart=time.time()
    total_tima=0
    results_path='/home/sunjc0306/HEI-Human/Pin'
    for img_path in img_path_list:
        # img=cv2.imread(img_path)
        # img_path="/media/sunjc0306/KESU/dataset/pingterest/ReImage/%6d.jpg"%(count)
        # cv2.imwrite(img_path,img)
        # img_name = img_path.split("/")[-1].rsplit(".", 1)[0]
        # save_path = '%s/%s_meshRefined.obj' % (results_path, img_name)
        # if os.path.exists(save_path):
        #     continue
        singTime=test.demo(img_path,results_path,save_inter=False)
        count += 1
        hrsPassed = (time.time() - timeStart) / 3600.
        hrsEachIter = hrsPassed / count
        numItersRemain = len(img_path_list) - count
        hrsRemain = numItersRemain * hrsEachIter  # hours that remain
        minsRemain = hrsRemain * 60.  # minutes that remain
        img_name = img_path.split("/")[-1].rsplit(".", 1)[0]
        total_tima+=singTime
        print("%s| %.4f | inference: %03d-%03d-%03d | remains %.3f m(s) ......" % (img_name,singTime,0,count,len(img_path_list)-count, minsRemain))
    print('The cost of time :',total_tima/count)

