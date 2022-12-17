# -*- coding: utf-8 -*-
from yacs.config import CfgNode as CN
import os

_C = CN(new_allowed=True)

# needed by trainer
_C.batch_size=1
_C.color_loss_type="l1"
_C.continue_train=False
_C.debug=False
_C.deepVoxels_c_len=8
_C.deepVoxels_c_len_intoLateFusion=8
_C.deepVoxels_fusion=None
_C.discriminator_accuracy_update_threshold=0.8
_C.displacment=0.0722
_C.epoch_offline_len=15
_C.epoch_range=[0, 15]
_C.freq_plot=10
_C.freq_save=50
_C.freq_save_ply=100
_C.gamma=0.1
_C.gen_test_mesh=False
_C.gen_train_mesh=False
_C.gpu_id='1'
_C.gpu_ids='0,1'
_C.hg_down="ave_pool"
_C.hourglass_dim=256
_C.img_path=None
_C.learning_rate=0.001
_C.learning_rateC=0.001
_C.learning_rate_3d_gan=1e-05
_C.loadSize=512
_C.load_checkpoint_path=None
_C.load_from_multi_GPU_shape=False
_C.load_netC_checkpoint_path=None
_C.load_netG_checkpoint_path="/home/sunjc0306/HEI-Human/checkpoints/netG_epoch_2_7992"
_C.load_netV_checkpoint_path="/home/sunjc0306/geopifu/checkpoints/GeoPIFu_coarse/netV_epoch_6_28991"
_C.mask_path=None
_C.mlp_dim=[257, 1024, 512, 256, 128, 1]
_C.mlp_dim_3d=[56, 256, 128, 1]
_C.mlp_dim_color=[513, 1024, 512, 256, 128, 3]
_C.mlp_dim_joint=[0, 256, 128, 1]
_C.multiRanges_deepVoxels=False
_C.no_gen_mesh=False
_C.no_num_eval=False
_C.no_residual=False
_C.norm="group"
_C.norm_color="group"
_C.num_epoch=100
_C.num_gen_mesh_test=1
_C.num_hourglass=2
_C.num_sample_color=0
_C.num_sample_inout=5000
_C.num_samples=70000
_C.num_skip_frames=1
_C.num_stack=4
_C.num_threads=1
_C.num_views=1
_C.occupancy_loss_type="mse"
_C.online_sampling=False
_C.pin_memory=False
_C.random_flip=False
_C.random_multiview=False
_C.random_scale=False
_C.random_trans=False
_C.recover_dim=False
_C.resolution=256
_C.resolution_x=171
_C.resolution_y=256
_C.resolution_z=171
_C.results_path="./results"
_C.resume_epoch=-1
_C.resume_iter=-1
_C.resume_name="example"
_C.sampleType="occu_sigma3.5_pts5k"
_C.schedule=[60, 80]
_C.serial_batches=False
_C.shuffle_train_test_ids=False
_C.sigma=5.0
_C.skip_hourglass=False
_C.splitIdx=0
_C.splitNum=8
_C.test_folder_path=None
_C.totalNumFrame=108720
_C.use_tanh=False
_C.val_test_error=False
_C.val_train_error=False
_C.vrn_net_input_height=384
_C.vrn_net_input_width=256
_C.vrn_num_hourglass=2
_C.vrn_num_modules=4
_C.vrn_occupancy_loss_type="ce"
_C.weight_3d_gan_gen=15.0
_C.weight_occu=1000.0
_C.weight_rgb_recon=200.0
_C.z_size=200.0
_C.num_epochs=10
_C.aug_alstd= 0.0
_C.aug_blur= 0.0
_C.aug_bri= 0.0
_C.aug_con= 0.0
_C.aug_hue= 0.0
_C.aug_sat= 0.0
_C.soft_onehot=False
_C.summary_steps=100
def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


# Alternatively, provide a way to import the defaults as
# a global singleton:
cfg = _C  # users can `from config import cfg`

# cfg = get_cfg_defaults()
# cfg.merge_from_file('../configs/example.yaml')

# # Now override from a list (opts could come from the command line)
# opts = ['dataset.root', '../data/XXXX', 'learning_rate', '1e-2']
# cfg.merge_from_list(opts)


def update_cfg(cfg_file):
    # cfg = get_cfg_defaults()
    _C.merge_from_file(cfg_file)
    # return cfg.clone()
    return _C


def parse_args(args):
    cfg_file = args.cfg_file
    if args.cfg_file is not None:
        cfg = update_cfg(args.cfg_file)
    else:
        cfg = get_cfg_defaults()

    # if args.misc is not None:
    #     cfg.merge_from_list(args.misc)

    return cfg


def parse_args_extend(args):
    if args.resume:
        if not os.path.exists(args.log_dir):
            raise ValueError(
                'Experiment are set to resume mode, but log directory does not exist.'
            )

        # load log's cfg
        cfg_file = os.path.join(args.log_dir, 'cfg.yaml')
        cfg = update_cfg(cfg_file)

        if args.misc is not None:
            cfg.merge_from_list(args.misc)
    else:
        parse_args(args)
