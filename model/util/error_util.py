from model.util.mesh_util import *
from .io_util import *
from tqdm import tqdm
import pdb  # pdb.set_trace()

def compute_acc(pred, gt, thresh=0.5):
    """
    input
        res         : (1, 1, n_in + n_out), res[0] are estimated occupancy probs for the query points
        label_tensor: (1, 1, n_in + n_out), float 1.0-inside, 0.0-outside

    return
        IOU, precision, and recall
    """

    # compute {IOU, precision, recall} based on the current query 3D points
    with torch.no_grad():

        vol_pred = pred > thresh
        vol_gt = gt > thresh

        union = vol_pred | vol_gt
        inter = vol_pred & vol_gt

        true_pos = inter.sum().float()

        union = union.sum().float()
        if union == 0:
            union = 1

        vol_pred = vol_pred.sum().float()
        if vol_pred == 0:
            vol_pred = 1

        vol_gt = vol_gt.sum().float()
        if vol_gt == 0:
            vol_gt = 1

        return true_pos / union, true_pos / vol_pred, true_pos / vol_gt
def calc_error_coarse(opt, net, cuda, dataset, num_tests):
    """
    return
        avg. {error, IoU, precision, recall} computed among num_test frames, each frame has e.g. 5000 query points for evaluation.
    """
    if num_tests > len(dataset):
        num_tests = len(dataset)
    with torch.no_grad():
        error_total_arr, IOU_arr, prec_arr, recall_arr = [], [], [], []
        for idx in tqdm(range(num_tests)):
            # retrieve data for one frame (or multi-view)
            data              = dataset[idx * len(dataset) // num_tests]
            voxel_tensor      = data['smplSemVoxels'].to(device=cuda)        # (V, C-3, H-512, W-512), RGB, 3x512x512 images, float -1. ~ 1., bg is all ZEROS not -1.
            meshVoxels_tensor = data['meshVoxels'].to(device=cuda) # (1, D-128, H-192, W-128), float 1.0-inside, 0.0-outside
            # reshape the data
            voxel_tensor            =            voxel_tensor.view(-1,  3,       voxel_tensor.shape[-3],       voxel_tensor.shape[-2],       voxel_tensor.shape[-1]) # (V,      C-3, H-512, W-512)
            meshVoxels_tensor       =       meshVoxels_tensor.view(-1, 1, meshVoxels_tensor.shape[-3],  meshVoxels_tensor.shape[-2],  meshVoxels_tensor.shape[-1]) # (1, 1, D-128, H-192, W-128)
            forward_return_dict = net.forward(images=voxel_tensor, labels=meshVoxels_tensor,)
            pred_occ                           = forward_return_dict["pred_occ"]
            error                              = forward_return_dict["error"].mean().item()
            # compute errors {IOU, prec, recall} based on the current set of query 3D points
            IOU, prec, recall = compute_acc(pred_occ, meshVoxels_tensor) # R, R, R
            # print(
            #     '{0}/{1} | Error: {2:06f} IOU: {3:06f} prec: {4:06f} recall: {5:06f}'
            #         .format(idx, num_tests, error.item(), IOU.item(), prec.item(), recall.item()))
            error_total_arr.append(error)
            IOU_arr.append(IOU.item())
            prec_arr.append(prec.item())
            recall_arr.append(recall.item())

    return np.average(error_total_arr), np.average(IOU_arr), np.average(prec_arr), np.average(recall_arr)


def calc_error_refine(opt, net, cuda, dataset, num_tests):
    """
    return
        avg. {error, IoU, precision, recall} computed among num_test frames, each frame has e.g. 5000 query points for evaluation.
    """
    if num_tests > len(dataset):
        num_tests = len(dataset)
    with torch.no_grad():
        erorr_arr, IOU_arr, prec_arr, recall_arr = [], [], [], []
        for idx in tqdm(range(num_tests)):

            # retrieve data for one frame (or multi-view)
            data = dataset[idx * len(dataset) // num_tests]
            image_tensor  = data['img'].to(device=cuda)                  # (num_views, C, W, H) for 3x512x512 images, float -1. ~ 1.
            calib_tensor  = data['calib'].to(device=cuda)                # (num_views, 4, 4) calibration matrix
            sample_tensor = data['samples'].to(device=cuda).unsqueeze(0) # (1, 3, n_in + n_out), float XYZ coords are inside the 3d-volume of [self.B_MIN, self.B_MAX]
            label_tensor = data['labels'].to(device=cuda).unsqueeze(0)   # (1, 1, n_in + n_out), float 1.0-inside, 0.0-outside
            deepVoxels_tensor = torch.zeros([label_tensor.shape[0]], dtype=torch.int32).to(device=cuda) # small dummy tensors
            deepVoxels_tensor = data["deepVoxels"].to(device=cuda)[None,:] # (B=1,C=8,D=32,H=48,W=32), np.float32, all >= 0.

            # forward pass
            res, error = net.forward(image_tensor, sample_tensor, calib_tensor, labels=label_tensor, deepVoxels=deepVoxels_tensor) # (1, 1, n_in + n_out), R
            if len(opt.gpu_ids) > 1: error = error.mean()

            # compute errors {IOU, prec, recall} based on the current set of query 3D points
            IOU, prec, recall = compute_acc(res, label_tensor) # R, R, R

            # print(
            #     '{0}/{1} | Error: {2:06f} IOU: {3:06f} prec: {4:06f} recall: {5:06f}'
            #         .format(idx, num_tests, error.item(), IOU.item(), prec.item(), recall.item()))
            erorr_arr.append(error.item())
            IOU_arr.append(IOU.item())
            prec_arr.append(prec.item())
            recall_arr.append(recall.item())

    return np.average(erorr_arr), np.average(IOU_arr), np.average(prec_arr), np.average(recall_arr)

