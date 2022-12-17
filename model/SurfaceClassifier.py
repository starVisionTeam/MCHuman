import torch
import torch.nn as nn
import torch.nn.functional as F
class SurfaceClassifier(nn.Module):
    def __init__(self, filter_channels, num_views=1, no_residual=True, last_op=None, opt=None):
        """
        input
            filter_channels: default is [257, 1024, 512, 256, 128, 1]
            no_residual    : default is False
        """
        super(SurfaceClassifier, self).__init__()

        self.filters = [] # length is filter_channels-1, default is 5   
        self.num_views = num_views
        self.no_residual = no_residual
        filter_channels = filter_channels # default: [257, 1024, 512, 256, 128, 1]
        self.last_op = last_op
        self.opt = opt

        # default is False
        if self.no_residual:
            for l in range(0, len(filter_channels) - 1):
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    1))
                self.add_module("conv%d" % l, self.filters[l])
        else:

            for l in range(0, len(filter_channels) - 1):
                if 0 != l:
                    self.filters.append(
                        nn.Conv1d(
                            filter_channels[l] + filter_channels[0],
                            filter_channels[l + 1],
                            1))
                else:
                    self.filters.append(nn.Conv1d(
                        filter_channels[l],
                        filter_channels[l + 1],
                        1))

                self.add_module("conv%d" % l, self.filters[l])

    def forward(self, feature):
        '''
        Input
            feature: (B * num_views, opt.hourglass_dim+1, n_in+n_out)
        
        Return
            y: (B, 1, n_in+n_out), num_views are canceled out by mean pooling
        '''

        y = feature
        tmpy = feature
        for i, f in enumerate(self.filters):

            # default is False
            if self.no_residual:
                # y = f(y)
                y = self._modules["conv%d"%(i)](y) if len(self.opt.gpu_ids) > 1 else f(y)
            else:
                # y = f(
                #     y if i == 0
                #     else torch.cat([y, tmpy], 1)
                # ) # with skip connections from feature
                y = self._modules["conv%d"%(i)]( y if i==0 else torch.cat([y, tmpy],1) ) if len(self.opt.gpu_ids) > 1 else f( y if i == 0 else torch.cat([y, tmpy],1) )

            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)

            if self.num_views > 1 and i == len(self.filters) // 2:
                y = y.view(
                    -1, self.num_views, y.shape[1], y.shape[2]
                ).mean(dim=1)
                tmpy = feature.view(
                    -1, self.num_views, feature.shape[1], feature.shape[2]
                ).mean(dim=1)

        if self.last_op:
            y = self.last_op(y)

        return y
