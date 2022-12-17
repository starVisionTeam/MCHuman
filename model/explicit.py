from .HGFilters import *
from model.util.net_util import init_net
# from .our_util import *


class ExplicitNet(nn.Module):

    def __init__(self, opt):
        super(ExplicitNet, self).__init__()

        # ----- init. -----

        self.name = 'ExplicitNet'
        self.opt = opt
        self.im_feat_list = []  # a list of deep voxel features
        self.intermediate_preds_list = []  # a list of estimated occupancy grids

        # ----- generate deep voxels -----
        if True:
            self.ResNet3D = generate_model(18)
            c_len_deepvoxels = 8
            for hg_module in range(self.opt.vrn_num_modules):  # default: 4

                self.add_module('m' + str(hg_module),
                                VolumeHourGlass(1, opt.num_hourglass, 64, self.opt.norm, self.opt.upsample_mode))

                self.add_module('top_m_' + str(hg_module), BasicBlock(64, 64))
                self.add_module('conv_last' + str(hg_module), nn.Conv3d(64, 128, kernel_size=1, stride=1, padding=0))
                if self.opt.norm == 'batch':
                    self.add_module('bn_end' + str(hg_module), nn.BatchNorm3d(128))
                elif self.opt.norm == 'group':
                    self.add_module('bn_end' + str(hg_module), nn.GroupNorm(32, 128))

                self.add_module("branch_out_3d_unet" + str(hg_module), Unet3D(c_len_in=256,
                                                                              c_len_out=c_len_deepvoxels))  # in-(BV,8,32,48,32), out-(BV,8,32,48,32)
                self.add_module('l' + str(hg_module),
                                nn.Conv3d(128, 256, kernel_size=1, stride=1, padding=0))
                if hg_module < self.opt.vrn_num_modules - 1:
                    self.add_module('bl' + str(hg_module), nn.Conv3d(128, 64, kernel_size=1, stride=1, padding=0))
                    self.add_module('cl' + str(hg_module),
                                    nn.Conv3d(8, 64, kernel_size=1, stride=1, padding=0))
        # ----- occupancy classification from deep voxels -----
        if True:
            self.conv3d_cls_1 = nn.Sequential(Conv3dSame(c_len_deepvoxels, c_len_deepvoxels, kernel_size=3, bias=False),
                                              nn.BatchNorm3d(c_len_deepvoxels, affine=True), nn.LeakyReLU(0.2, True))
            self.conv3d_cls_2 = nn.Sequential(Conv3dSame(c_len_deepvoxels, 1, kernel_size=1, bias=True), nn.Sigmoid())

        # weights initialization for conv, fc, batchNorm layers
        init_net(self)

    def get_error(self, labels):
        '''
        Hourglass has its own intermediate supervision scheme
        '''

        # init.
        error_dict = {}

        # accumulate errors from all the latent feature layers
        count = 0
        error = 0.
        for preds in self.intermediate_preds_list:

            # occupancy CE loss, random baseline is: (1000. * -np.log(0.5) / 2. == 346.574), optimal is: 0.
            w = 0.7
            error += self.opt.weight_occu * (-w * torch.mean(
                labels * torch.log(preds + 1e-8))  # preds: (BV,1,128,192,128), labels: (BV,1,128,192,128)
                                             - (1 - w) * torch.mean((1 - labels) * torch.log(1 - preds + 1e-8))
                                             )  # R



            # update count
            count += 1

        # average loss over different latent feature layers
        error_dict["error"] = error / count

        return error_dict

    def filter(self, images):

        images = self.ResNet3D(images)

        previous=images
        self.im_feat_list = []
        for i in range(self.opt.vrn_num_modules):  # default: 4

            hg = self._modules['m' + str(i)](previous)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)](self._modules['conv_last' + str(i)](ll)), True)


            # Predict heatmaps
            tmp_out = self._modules['l' + str(i)](ll)  # (BV,256,48,32)
            # assert (tmp_out.shape[1] % tmp_out.shape[-1] == 0)
            # tmp_out = tmp_out.view(tmp_out.shape[0], -1, tmp_out.shape[-1], tmp_out.shape[-2],
            #                        tmp_out.shape[-1])  # (BV,8,32,48,32)
            tmp_out = self._modules['branch_out_3d_unet' + str(i)](tmp_out)  # (BV,8,32,48,32)
            if self.training:

                self.im_feat_list.append(tmp_out)
            else:

                if i == (self.opt.vrn_num_modules - 1): self.im_feat_list.append(tmp_out)
            # tmp_out = tmp_out.view(tmp_out.shape[0], -1, tmp_out.shape[-2], tmp_out.shape[-1])  # (BV,256,48,32)

            if i < (self.opt.vrn_num_modules - 1):
                ll = self._modules['bl' + str(i)](ll)
                # tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll +  self._modules['cl' + str(i)](tmp_out)

    def est_occu(self):

        # init.
        self.intermediate_preds_list = []

        # for each level of deep voxels inside the stack-hour-glass networks
        max_count = len(self.im_feat_list) - 1
        count = 0
        for im_feat in self.im_feat_list:

            # upsampling x4 the deep voxels
            # (BV,8,32,48,32), (BV,8,32,48,32), (BV,8,32,48,32), (BV,8,32,48,32)             | upsampling x4                                | (BV,8,128,192,128), (BV,8,128,192,128), (BV,8,128,192,128), (BV,8,128,192,128)
            deepVoxels_upsampled = F.interpolate(im_feat, scale_factor=4, mode='trilinear',
                                                 align_corners=True)  # (BV,8,128,192,128)

            # ----- occupancy classification from deep voxels -----
            # (BV,8,128,192,128), (BV,8,128,192,128), (BV,8,128,192,128), (BV,8,128,192,128) | conv3d(i8,o8,k3,s1,nb), BN3d, LeakyReLU(0.2) | (BV,8,128,192,128), (BV,8,128,192,128), (BV,8,128,192,128), (BV,8,128,192,128)
            # (BV,8,128,192,128), (BV,8,128,192,128), (BV,8,128,192,128), (BV,8,128,192,128) | conv3d(i8,o1,k1,s1,b), sigmoid               | (BV,1,128,192,128), (BV,1,128,192,128), (BV,1,128,192,128), (BV,1,128,192,128)
            if True:

                # conv3d(i8,o8,k3,s1,nb), BN3d, LeakyReLU(0.2)
                pred_final = self.conv3d_cls_1(deepVoxels_upsampled)  # (BV,8,128,192,128)

                # conv3d(i8,o1,k1,s1,b), sigmoid
                pred_final = self.conv3d_cls_2(pred_final)  # (BV,1,128,192,128)
                self.intermediate_preds_list.append(pred_final)
            # update count
            count += 1

    def get_preds(self):

        return self.intermediate_preds_list[-1]  # BCDHW, (BV,1,128,192,128), est. occupancy



    def forward(self, images, labels=None):

        # init
        return_dict = {}
        self.labels = labels

        # compute deep voxels
        self.filter(images=images)

        # estimate occupancy grids, (and render prediction)
        self.est_occu()

        # get the estimated_occupancy
        return_dict["pred_occ"] = self.get_preds()  # BCDHW, (BV,1,128,192,128), est. occupancy
        # compute occupancy errors
        error = self.get_error(
            labels=labels)  # R, the mean loss value over all latent feature maps of the stack-hour-glass network
        return_dict.update(error)
        # return: estimated mesh voxels, error, (and render_rgb, pseudo_inverseDepth, error_view_render), (and error_3d_gan_generator, error_3d_gan_discriminator_fake, error_3d_gan_discriminator_real)
        return return_dict


if __name__ == '__main__':
    # get options
    from model.options import BaseOptions

    opt = BaseOptions().parse()
    vrn = ExplicitNet(opt).cuda()
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in vrn.parameters())))
    # ResNet = generate_model(10).cuda()
    # print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in ResNet.parameters())))
    # img = torch.rand([4, 3, 128, 192, 128]).cuda()
    # print(ResNet(img).shape)
    # ResNet(img)
    images = torch.rand([2, 3, 128, 192, 128]).cuda()
    labels = torch.rand([2, 1, 128, 192, 128]).cuda()  # torch.Size([4, 1, 128, 192, 128])
    print(vrn(images, labels))






