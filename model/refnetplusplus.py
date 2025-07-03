
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#####################################
############ Camera #################
#####################################

class Bottleneck_camera(nn.Module):

    def __init__(self, in_planes, planes, stride=1, downsample=None, expansion=4):
        super(Bottleneck_camera, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(expansion * planes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = F.relu(residual + out)
        return out


class warmupblock(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size=1, use_bn=True):
        super(warmupblock, self).__init__()
        self.use_bn = use_bn

        self.conv1 = nn.Conv2d(in_layer, out_layer, kernel_size,
                               stride=(1, 1), padding=1, bias=(not use_bn))

        self.bn1 = nn.BatchNorm2d(out_layer)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        if self.use_bn:
            x1 = self.bn1(x1)
        x = self.relu(x1)
        return x


class FPN_BackBone_camera(nn.Module):

    def __init__(self, num_block, channels, block_expansion, use_bn=True):
        super(FPN_BackBone_camera, self).__init__()
        self.block_expansion = block_expansion
        self.use_bn = use_bn

        self.warmup = warmupblock(3, 16, kernel_size=3, use_bn=True)
        self.in_planes = 16

        self.conv = nn.Conv2d(self.in_planes, self.in_planes, kernel_size=3, stride=1,
                              padding=1, bias=False)
        self.bn = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=False)

        # Residuall blocks
        self.block1 = self._make_layer(Bottleneck_camera, planes=channels[0], num_blocks=num_block[0])
        self.block2 = self._make_layer(Bottleneck_camera, planes=channels[1], num_blocks=num_block[1])
        self.block3 = self._make_layer(Bottleneck_camera, planes=channels[2], num_blocks=num_block[2])
        self.block4 = self._make_layer(Bottleneck_camera, planes=channels[3], num_blocks=num_block[3])

    def forward(self, x):
        x = self.warmup(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        # Backbone
        features = {}
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        features['x0'] = x
        features['x1'] = x1
        features['x2'] = x2
        features['x3'] = x3
        features['x4'] = x4

        return features

    def _make_layer(self, block, planes, num_blocks):
        if self.use_bn:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * self.block_expansion,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(planes * self.block_expansion)
            )
        else:
            downsample = nn.Conv2d(self.in_planes, planes * self.block_expansion,
                                   kernel_size=1, stride=2, bias=True)

        layers = []
        layers.append(block(self.in_planes, planes, stride=2, downsample=downsample, expansion=self.block_expansion))
        self.in_planes = planes * self.block_expansion
        for i in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, stride=1, expansion=self.block_expansion))
            self.in_planes = planes * self.block_expansion
        return nn.Sequential(*layers)  # this *layers will unpack the list


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class upsample(nn.Module):

    def __init__(self, if_deconv, channels=None):
        super(upsample, self).__init__()
        if if_deconv:
            self.upsample = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1, bias=False)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.upsample(x)

        return x


class double_conv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class encoder_after_fpn(nn.Module):

    def __init__(self):
        super(encoder_after_fpn, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 3, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.mu_dec = nn.Linear(1024, 512)
        self.logvar_dec = nn.Linear(1024, 512)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 1024)
        mu = self.mu_dec(x)
        logvar = self.logvar_dec(x)

        return mu, logvar


class decoder_conv(nn.Module):
    def __init__(self, if_deconv):
        super(decoder_conv, self).__init__()

        # self.up1 = upsample(if_deconv=if_deconv, channels=64)
        # self.conv1 = double_conv(64, 64)
        self.up2 = upsample(if_deconv=if_deconv, channels=8)
        self.conv2 = double_conv(8, 8)
        self.up3 = upsample(if_deconv=if_deconv, channels=8)
        self.conv3 = double_conv(8, 8)

        self.up5 = upsample(if_deconv=if_deconv, channels=136)
        self.conv5 = double_conv(136, 64)
        self.up6 = upsample(if_deconv=if_deconv, channels=128)
        self.conv6 = double_conv(128, 64)

        self.up4 = upsample(if_deconv=if_deconv, channels=64)
        self.conv4 = double_conv(64, 32)

        self.L1 = nn.Conv2d(120, 64, kernel_size=1, stride=1, padding=0)
        self.L2 = nn.Conv2d(68, 64, kernel_size=1, stride=1, padding=0)

        self.L3 = nn.Conv2d(60, 32, kernel_size=1, stride=1, padding=0)
        self.L4 = nn.Conv2d(34, 32, kernel_size=1, stride=1, padding=0)

        # this is basically the segmentation head
        # self.conv_out1 = nn.Conv2d(128, 64, 3, padding=1)
        # self.conv_out2 = nn.Conv2d(64, 1, 3, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x, features):
        x = x.view(-1, 8, 8, 8)
        # x = self.up1(x)
        # x = self.conv1(x)
        x = self.up2(x)
        x = self.conv2(x)
        x = self.up3(x)
        x = self.conv3(x)

        T3 = features['x3'].transpose(1, 3)
        T3 = self.L3(T3)
        T3 = T3.transpose(1, 3)
        T3 = T3.transpose(1, 2)
        T3 = self.L4(T3)
        T3 = T3.transpose(1, 2)

        T2 = features['x2'].transpose(1, 3)
        T2 = self.L1(T2)
        T2 = T2.transpose(1, 3)
        T2 = T2.transpose(1, 2)
        T2 = self.L2(T2)
        T2 = T2.transpose(1, 2)

        x = torch.cat((x, T3), axis = 1)

        x = self.up5(x)
        x = self.conv5(x)

        x = torch.cat((x, T2), axis=1)

        x = self.up6(x)
        x = self.conv6(x)

        x = self.up4(x)
        x = self.conv4(x)

        return x

class cameraonly_perspective(nn.Module):
    def __init__(self, channels_bev, blocks):

        super(cameraonly_perspective, self).__init__()

        self.FPN = FPN_BackBone_camera(num_block=blocks, channels=channels_bev, block_expansion=2, use_bn=True)
        self.encoder_afterf_fpn = encoder_after_fpn()
        self.decoder = decoder_conv(if_deconv=True)

    def reparameterize(self, is_training, mu, logvar):
        if is_training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, is_training):
        features = self.FPN(x)
        mu, logvar = self.encoder_afterf_fpn(features['x4'])
        z = self.reparameterize(is_training, mu, logvar)
        cam_decoded = self.decoder(z, features)
        return cam_decoded


class cam_only_decoderskip(nn.Module):
    def __init__(self, channels_bev, blocks, detection_head, segmentation_head):
        super(cam_only_decoderskip, self).__init__()

        self.detection_head = detection_head
        self.segmentation_head = segmentation_head
        self.cameraencdec = cameraonly_perspective(channels_bev=channels_bev, blocks=blocks)

    def forward(self, cam_inputs, is_training):

        decoder_output = self.cameraencdec(cam_inputs, is_training)

        return decoder_output

####################################
############ Radar #################
####################################

NbTxAntenna = 12
NbRxAntenna = 16
NbVirtualAntenna = NbTxAntenna * NbRxAntenna

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

class Detection_Header(nn.Module):

    def __init__(self,config, use_bn=True, reg_layer=2):
        super(Detection_Header, self).__init__()

        self.use_bn = use_bn
        self.reg_layer = reg_layer
        self.config = config
        bias = not use_bn

        if config['model']['DetectionHead'] == 'True':
            self.conv1 = conv3x3(288, 144, bias=bias)
            self.bn1 = nn.BatchNorm2d(144)
            self.conv2 = conv3x3(144, 96, bias=bias)
            self.bn2 = nn.BatchNorm2d(96)

            self.conv3 = conv3x3(96, 96, bias=bias)
            self.bn3 = nn.BatchNorm2d(96)
            self.conv4 = conv3x3(96, 96, bias=bias)
            self.bn4 = nn.BatchNorm2d(96)

            self.clshead = conv3x3(96, 1, bias=True)
            self.reghead = conv3x3(96, reg_layer, bias=True)

    def forward(self, x):

        if self.config['model']['DetectionHead'] == 'True':
            x = self.conv1(x)
            if self.use_bn:
                x = self.bn1(x)
            x = self.conv2(x)
            if self.use_bn:
                x = self.bn2(x)
            x = self.conv3(x)
            if self.use_bn:
                x = self.bn3(x)
            x = self.conv4(x)
            if self.use_bn:
                x = self.bn4(x)

            cls = torch.sigmoid(self.clshead(x))
            reg = self.reghead(x)

            return torch.cat([cls, reg], dim=1)

class Bottleneck(nn.Module):

    def __init__(self, in_planes, planes, stride=1, downsample=None, expansion=4):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(expansion * planes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = F.relu(residual + out)
        return out


class MIMO_PreEncoder(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size=(1, 12), dilation=(1, 16), use_bn=False):
        super(MIMO_PreEncoder, self).__init__()
        self.use_bn = use_bn

        self.conv = nn.Conv2d(in_layer, out_layer, kernel_size,
                              stride=(1, 1), padding=0, dilation=dilation, bias=(not use_bn))

        self.bn = nn.BatchNorm2d(out_layer)
        self.padding = int(NbVirtualAntenna / 2)

    def forward(self, x):
        width = x.shape[-1]
        x = torch.cat([x[..., -self.padding:], x, x[..., :self.padding]], axis=3)
        x = self.conv(x)
        x = x[..., int(x.shape[-1] / 2 - width / 2):int(x.shape[-1] / 2 + width / 2)]

        if self.use_bn:
            x = self.bn(x)
        return x


class FPN_BackBone(nn.Module):

    def __init__(self, num_block, channels, block_expansion, mimo_layer, use_bn=True):
        super(FPN_BackBone, self).__init__()

        self.block_expansion = block_expansion
        self.use_bn = use_bn

        # pre processing block to reorganize MIMO channels
        self.pre_enc = MIMO_PreEncoder(32, mimo_layer,
                                       kernel_size=(1, NbTxAntenna),
                                       dilation=(1, NbRxAntenna),
                                       use_bn=True)

        self.in_planes = mimo_layer

        self.conv = conv3x3(self.in_planes, self.in_planes)
        self.bn = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)

        # Residuall blocks
        self.block1 = self._make_layer(Bottleneck, planes=channels[0], num_blocks=num_block[0])
        self.block2 = self._make_layer(Bottleneck, planes=channels[1], num_blocks=num_block[1])
        self.block3 = self._make_layer(Bottleneck, planes=channels[2], num_blocks=num_block[2])
        self.block4 = self._make_layer(Bottleneck, planes=channels[3], num_blocks=num_block[3])

    def forward(self, x):

        x = self.pre_enc(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        # Backbone
        features = {}
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        features['x0'] = x
        features['x1'] = x1
        features['x2'] = x2
        features['x3'] = x3
        features['x4'] = x4

        return features

    def _make_layer(self, block, planes, num_blocks):
        if self.use_bn:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * self.block_expansion,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(planes * self.block_expansion)
            )
        else:
            downsample = nn.Conv2d(self.in_planes, planes * self.block_expansion,
                                   kernel_size=1, stride=2, bias=True)

        layers = []
        layers.append(block(self.in_planes, planes, stride=2, downsample=downsample, expansion=self.block_expansion))
        self.in_planes = planes * self.block_expansion
        for i in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, stride=1, expansion=self.block_expansion))
            self.in_planes = planes * self.block_expansion
        return nn.Sequential(*layers)


class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            out = self.downsample(out)

        return out


class RangeAngle_Decoder(nn.Module):
    def __init__(self, ):
        super(RangeAngle_Decoder, self).__init__()

        # Top-down layers
        self.deconv4 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0))

        self.conv_block4 = BasicBlock(48, 128)
        self.deconv3 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0))
        self.conv_block3 = BasicBlock(192, 256)

        self.L3 = nn.Conv2d(192, 224, kernel_size=1, stride=1, padding=0)
        self.L2 = nn.Conv2d(160, 224, kernel_size=1, stride=1, padding=0)

    def forward(self, features):
        T4 = features['x4'].transpose(1, 3)
        T3 = self.L3(features['x3']).transpose(1, 3)
        T2 = self.L2(features['x2']).transpose(1, 3)

        S4 = torch.cat((self.deconv4(T4), T3), axis=1)
        S4 = self.conv_block4(S4)

        S43 = torch.cat((self.deconv3(S4), T2), axis=1)
        out = self.conv_block3(S43)

        return out


class PolarSegFusionNet(nn.Module):
    def __init__(self, mimo_layer, channels, blocks, detection_head, segmentation_head, config, regression_layer):
        super(PolarSegFusionNet, self).__init__()

        self.detection_head = detection_head
        self.segmentation_head = segmentation_head

        self.FPN = FPN_BackBone(num_block=blocks, channels=channels, block_expansion=4, mimo_layer=mimo_layer,
                                use_bn=True)
        self.RA_decoder = RangeAngle_Decoder()

    def forward(self, x):

        features = self.FPN(x)
        RA = self.RA_decoder(features)

        return RA

##################################
############FUSION################
##################################

class refnetplusplus(nn.Module):
    def __init__(self, mimo_layer, channels, channels_bev, blocks, detection_head, segmentation_head, config, regression_layer=2):
        super(refnetplusplus, self).__init__()

        self.detection_head = detection_head
        self.segmentation_head = segmentation_head

        self.radarenc = PolarSegFusionNet(mimo_layer=mimo_layer, channels=channels, blocks=blocks, detection_head=detection_head, segmentation_head=segmentation_head, config=config, regression_layer=regression_layer )
        self.cameraenc = cam_only_decoderskip(channels_bev=channels_bev, blocks=blocks, detection_head=detection_head, segmentation_head=segmentation_head)

        if (self.detection_head == "True"):
            self.detection_header = Detection_Header(config=config, reg_layer=regression_layer)

        if (self.segmentation_head == "True"):
            self.freespace = nn.Sequential(BasicBlock(288, 128), BasicBlock(128, 64), nn.Conv2d(64, 1, kernel_size=1))

    def forward(self, cam_inputs, ra_inputs, is_training):

        out = {'Detection':[],'Segmentation':[]}

        RA_decoded = self.radarenc(ra_inputs)

        BEV_decoded = self.cameraenc(cam_inputs, is_training)

        if (self.detection_head == "True"):
            x_det = F.interpolate(BEV_decoded, (128, 224))
            out_fused_det = torch.cat((RA_decoded, x_det), axis=1)
            out['Detection'] = self.detection_header(out_fused_det)

        if (self.segmentation_head == "True"):
            x_seg = F.interpolate(BEV_decoded, (256, 224))
            y_seg = F.interpolate(RA_decoded, (256, 224))
            out_fused_seg = torch.cat((y_seg, x_seg), axis=1)
            out['Segmentation'] = self.freespace(out_fused_seg)

        return out