import numpy as np
import torch
from model.util.sdf import create_grid, eval_grid_octree, eval_grid
from skimage import measure
from PIL import Image
from .io_util import save_volume,save_obj_mesh_with_color
from .geometry import verts_canonization,index
import pdb
def gen_mesh_coarse(opt, net, cuda, data, save_path, also_generate_mesh_from_gt_voxels=True):
    # init,
    save_path_png, save_path_gt_obj = None, None
    visualCheck_0 = False

    # retrieve and reshape the data for one frame (or multi-view)
    Voxels_tensor = data['smplSemVoxels'].to(
        device=cuda)  # (V, C-3,D-128 H-192, W-128)
    # Voxels_tensor = Voxels_tensor.view(-1,3, Voxels_tensor.shape[-3], Voxels_tensor.shape[-2],
    #                                  Voxels_tensor.shape[-1])  # (V, C-3, H-512, W-512)
    Voxels_tensor=Voxels_tensor.permute((3, 2, 1, 0)).unsqueeze(0)
    img = data['img'].to(
        device=cuda)  # (V, C-3,D-128 H-192, W-128)
    net.filter(Voxels_tensor)
    try:
        # ----- save the single-view/multi-view input image(s) of this data point -----
        if True:
            save_img_path = save_path[:-4] + '.png'
            save_img_list = []
            for v in range(img.shape[0]):
                save_img = (np.transpose(img[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :,
                           ::-1] * 255.0  # RGB -> BGR, (3,512,512), [0, 255]
                save_img_list.append(save_img)
            save_img = np.concatenate(save_img_list, axis=1)
            Image.fromarray(np.uint8(save_img[:, :, ::-1])).save(
                save_img_path)  # BGR -> RGB, cuz "PIL save RGB-array into proper-color.png"

            # update return path
            save_path_png = save_img_path

        # ----- save est. mesh with proj-color -----
        if True:

            # get est. occu.
            net.est_occu()
            pred_occ = net.get_preds()  # BCDHW, (V,1,128,192,128), est. occupancy
            pred_occ = pred_occ[0, 0].detach().cpu().numpy()  # DHW
            pred_occ = np.transpose(pred_occ, (2, 1, 0))  # WHD, XYZ
            if visualCheck_0:
                print("visualCheck_0: check the est voxels...")
                save_volume(pred_occ > 0.5,
                            fname="./sample_images/%s_est_mesh_voxels.obj" % (save_path[:-4].split("/")[-1]), dim_h=192,
                            dim_w=128, voxel_size=1. / 192.)
                pdb.set_trace()

            # est, marching cube
            vol = pred_occ  # WHD, XYZ
            verts, faces, normals, _ = measure.marching_cubes_lewiner(vol,
                                                                      level=0.5)  # https://scikit-image.org/docs/dev/api/skimage.measure.html#marching-cubes-lewiner
            verts = verts * 2.0  # this is only to match the verts_canonization function
            verts = verts_canonization(verts=verts, dim_w=pred_occ.shape[0], dim_h=pred_occ.shape[1])
            verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()  # (1, 3, N)
            xyz_tensor = verts_tensor * 2.0  # （1, 3, N） Tensor of xyz coordinates in the image plane of (-1,1) zone and of the-first-view
            uv = xyz_tensor[:, :2, :]  # (1, 2, N) for xy, float -1 ~ 1
            color = index(img[:1], uv).detach().cpu().numpy()[0].T  # (N, 3), RGB, float -1 ~ 1
            color = color * 0.5 + 0.5  # (N, 3), RGB, float 0 ~ 1
            save_obj_mesh_with_color(save_path, verts, faces, color)

        # ----- save marching cube mesh from gt. low-resolution mesh voxels -----
        if also_generate_mesh_from_gt_voxels:
            # get gt. occu.
            meshVoxels_tensor = data['meshVoxels'].to(
                device=cuda)  # (1, D-128, H-192, W-128), float 1.0-inside, 0.0-outside

            gt_occ = meshVoxels_tensor[0].detach().cpu().numpy()  # DHW
            gt_occ = np.transpose(gt_occ, (2, 1, 0))  # WHD, XYZ

            # gt, marching cube
            save_path = save_path[:-4] + '_GT_lowRes.obj'
            vol = gt_occ  # WHD, XYZ
            verts, faces, normals, _ = measure.marching_cubes_lewiner(vol,level=0.5)  # https://scikit-image.org/docs/dev/api/skimage.measure.html#marching-cubes-lewiner
            verts = verts * 2.0  # this is only to match the verts_canonization function
            verts = verts_canonization(verts=verts, dim_w=pred_occ.shape[0], dim_h=pred_occ.shape[1])
            verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()  # (1, 3, N)
            xyz_tensor = verts_tensor * 2.0  # （1, 3, N） Tensor of xyz coordinates in the image plane of (-1,1) zone and of the-first-view
            uv = xyz_tensor[:, :2, :]  # (1, 2, N) for xy, float -1 ~ 1
            color = index(img[:1], uv).detach().cpu().numpy()[0].T  # (N, 3), RGB, float -1 ~ 1
            color = color * 0.5 + 0.5  # (N, 3), RGB, float 0 ~ 1
            save_obj_mesh_with_color(save_path, verts, faces, color)
            # update return path
            save_path_gt_obj = save_path
    except Exception as e:
        print(e)
        print('Can not create marching cubes at this time.')
    # return paths
    return save_path_png, save_path_gt_obj
def reconstruction(net, cuda, calib_tensor, resolution_x, resolution_y, resolution_z, b_min, b_max, use_octree=False, num_samples=7000, transform=None, deepVoxels=None):
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param net: a BasePixImpNet object. call image filter beforehead.
    :param cuda: cuda device
    :param calib_tensor: calibration tensor, (num_views, 4, 4)
    :param resolution: resolution of the grid cell, 256
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param use_octree: whether to use octree acceleration, True
    :param num_samples: how many points to query each gpu iteration
    :return: marching cubes results.
    '''

    # First we create a grid by resolution and transforming matrix for grid coordinates to real world xyz
    # coords: WHD, XYZ, voxel-space converted to mesh-coords, (3, 256, 256, 256)
    # mat   : 4x4, {XYZ-scaling, trans} matrix from voxel-space to mesh-coords, by left Mul. with voxel-space idx tensor
    coords, mat = create_grid(resolution_x, resolution_y, resolution_z, b_min, b_max, transform=transform)

    # Then we define the lambda function for cell evaluation
    def eval_func(points):
        points  = np.expand_dims(points, axis=0)                   # (1,         3, num_samples)
        points  = np.repeat(points, net.num_views, axis=0)         # (num_views, 3, num_samples)
        samples = torch.from_numpy(points).to(device=cuda).float() # (num_views, 3, num_samples)
        net.query(points=samples, calibs=calib_tensor, deepVoxels=deepVoxels) # calib_tensor is (num_views, 4, 4)
        pred = net.get_preds()[0][0]                               # (num_samples,)
        return pred.detach().cpu().numpy()   

    # Then we evaluate the grid, use_octree default: True
    if use_octree:
        sdf = eval_grid_octree(coords, eval_func, num_samples=num_samples) # XYZ, WHD, (256, 256, 256), float 0. ~ 1. for occupancy
    else:
        sdf = eval_grid(coords, eval_func, num_samples=num_samples) # XYZ, WHD, (256, 256, 256), float 0. ~ 1. for occupancy
    # Finally we do marching cubes
    try:
        verts, faces, normals, values = measure.marching_cubes_lewiner(sdf, 0.5)
        # transform verts into world coordinate system
        verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4] # (3,N), convert verts from voxel-space into mesh-coords
        verts = verts.T # (N,3)
        return verts, faces, normals, values,sdf
    except:
        print('error cannot marching cubes')
        return -1
def gen_mesh_refine(opt, net, cuda, data, save_path, use_octree=True):

    image_tensor      = data['img'].to(device=cuda)   # (num_views, 3, 512, 512)
    calib_tensor      = data['calib'].to(device=cuda) # (num_views, 4, 4)
    deepVoxels_tensor = data["deepVoxels"].to(device=cuda) # (B=1,C=8,D=32,H=48,W=32), np.float32, all >= 0.
    # use hour-glass networks to extract image features
    net.filter(image_tensor)
    # the volume of query space
    b_min = data['b_min']
    b_max = data['b_max']
    try:
        # ----- save the single-view/multi-view input image(s) of this data point -----
        # save_img_path = save_path[:-4] + '.png'
        # save_img_list = []
        # for v in range(image_tensor.shape[0]):
        #     save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0 # RGB -> BGR, (3,512,512), [0, 255]
        #     save_img_list.append(save_img)
        # save_img = np.concatenate(save_img_list, axis=1)
        # Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path) # BGR -> RGB, cuz "PIL save RGB-array into proper-color.png"

        # ----- save mesh with proj-color -----

        # verts: (N, 3) in the mesh-coords, the same coord as data["samples"], N << 256*256*256
        # faces: (N, 3)
        verts, faces, _, _ ,sdf= reconstruction(net, cuda, calib_tensor, opt.resolution_x, opt.resolution_y, opt.resolution_z, b_min, b_max, use_octree=use_octree, deepVoxels=deepVoxels_tensor)
        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float() # (1, N, 3)
        xyz_tensor = net.projection(verts_tensor, calib_tensor[:1]) # （1, 3, N） Tensor of xyz coordinates in the image plane of (-1,1) zone and of the-first-view
        uv = xyz_tensor[:, :2, :] # (1, 2, N) for xy, float -1 ~ 1
        color = index(image_tensor[:1], uv).detach().cpu().numpy()[0].T # (N, 3), RGB, float -1 ~ 1
        color = color * 0.5 + 0.5 # (N, 3), RGB, float 0 ~ 1
        save_obj_mesh_with_color(save_path, verts, faces, color)
        return sdf
    except Exception as e:
        print(e)
        print('Can not create marching cubes at this time.')




