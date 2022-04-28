import numpy as np
import torch
from einops import rearrange
import torch.nn as nn
import torchsummary


class SR_Block(nn.Module):
    def __init__(self, in_channels, out_channels, bn_relu=True):
        super().__init__()
        conv_3x3_1_channels = int(np.round(out_channels / 6))
        conv_3x3_2_channels = int(np.round(out_channels / 3))
        conv_3x3_3_channels = int(np.round(out_channels / 2))
        self.bn_relu = None
        if bn_relu:
            self.bn_relu = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU())
        self.conv_1x1_1 = nn.Sequential(nn.Conv2d(in_channels, conv_3x3_1_channels, 1, 1, 0))
        self.conv_3x3_1 = nn.Sequential(nn.BatchNorm2d(conv_3x3_1_channels), nn.ReLU(), nn.Conv2d(conv_3x3_1_channels, conv_3x3_1_channels, 3, 1, 1))
        self.conv_3x3_2 = nn.Sequential(nn.BatchNorm2d(conv_3x3_1_channels * 2), nn.ReLU(), nn.Conv2d(conv_3x3_1_channels * 2, conv_3x3_2_channels, 3, 1, 1))
        self.conv_3x3_3 = nn.Sequential(nn.BatchNorm2d(conv_3x3_1_channels + conv_3x3_2_channels), nn.ReLU(), nn.Conv2d(conv_3x3_1_channels + conv_3x3_2_channels, conv_3x3_3_channels, 3, 1, 1))
        self.conv_1x1_2 = nn.Sequential(
            nn.BatchNorm2d(conv_3x3_1_channels + conv_3x3_2_channels + conv_3x3_3_channels),
            nn.ReLU(),
            nn.Conv2d(conv_3x3_1_channels + conv_3x3_2_channels + conv_3x3_3_channels, out_channels, 3, 1, 1),
        )
        self.skip_conv_1x1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, 1, 0))

    def forward(self, x):
        if self.bn_relu is not None:
            x = self.bn_relu(x)
        x_1 = self.conv_1x1_1(x)
        x_2 = self.conv_3x3_1(x_1)
        x_3 = self.conv_3x3_2(torch.cat([x_1, x_2], dim=1))
        x_4 = self.conv_3x3_3(torch.cat([x_2, x_3], dim=1))
        output = self.conv_1x1_2(torch.cat([x_2, x_3, x_4], dim=1)) + self.skip_conv_1x1(x)
        return output


class PGA(nn.Module):
    def __init__(self, en_channels, de_channels):
        super().__init__()
        self.conv_1x1_1 = nn.Sequential(nn.BatchNorm2d(en_channels), nn.ReLU(), nn.Conv2d(en_channels, (en_channels + de_channels) // 4, 1, 1, 0))
        self.conv_1x1_2 = nn.Sequential(nn.BatchNorm2d(de_channels), nn.ReLU(), nn.Conv2d(de_channels, (en_channels + de_channels) // 4, 1, 1, 0))
        self.conv_1x1_3 = nn.Sequential(
            nn.BatchNorm2d((en_channels + de_channels) // 4 * 2),
            nn.ReLU(),
            nn.Conv2d((en_channels + de_channels) // 4 * 2, (en_channels + de_channels) // 8, 1, 1, 0),
            nn.BatchNorm2d((en_channels + de_channels) // 8),
            nn.ReLU(),
            nn.Conv2d((en_channels + de_channels) // 8, 2, 1, 1, 0),
            nn.ReLU(),
            nn.Softmax(dim=1),
        )

    def forward(self, x_encoder, x_decoder):
        output = rearrange(self.conv_1x1_3(torch.cat([self.conv_1x1_1(x_encoder), self.conv_1x1_2(x_decoder)], dim=1)), 'b (n c) h w -> n b c h w', n=2)
        output = x_encoder * output[0] + x_decoder * output[1]
        return output


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn_relu = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU())
        self.aspp_block_1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        self.aspp_block_2 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.aspp_block_3 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, dilation=1)
        self.aspp_block_4 = nn.Conv2d(in_channels, out_channels, 3, 1, 2, dilation=2)
        self.aspp_block_5 = nn.Conv2d(in_channels, out_channels, 3, 1, 3, dilation=3)
        self.net = nn.Sequential(nn.BatchNorm2d(5 * out_channels), nn.ReLU(), nn.Conv2d(5 * out_channels, out_channels, 1, 1, 0))

    def forward(self, x):
        x = self.bn_relu(x)
        output = self.net(
            torch.cat(
                [
                    nn.Upsample(size=(x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False)(self.aspp_block_1(x)),
                    self.aspp_block_2(x),
                    self.aspp_block_3(x),
                    self.aspp_block_4(x),
                    self.aspp_block_5(x),
                ],
                dim=1,
            )
        )
        return output


class PG_SN(nn.Module):
    def __init__(self, config):
        super().__init__()
        size, in_channels, encoder_channels, decoder_channels, out_channels = (config['size'], config['in_channels'], config['encoder_channels'], config['decoder_channels'], config['out_channels'])
        assert size == 180, 'The image size does not meet the requirements, it must be 180.'
        self.pool_list = nn.ModuleList(
            [
                nn.Sequential(nn.BatchNorm2d(encoder_channels[0]), nn.ReLU(), nn.MaxPool2d(2, 2)),
                nn.Sequential(nn.BatchNorm2d(encoder_channels[1]), nn.ReLU(), nn.MaxPool2d(2, 2)),
                nn.Sequential(nn.BatchNorm2d(encoder_channels[2]), nn.ReLU(), nn.MaxPool2d(3, 2)),
                nn.Sequential(nn.BatchNorm2d(encoder_channels[3]), nn.ReLU(), nn.MaxPool2d(2, 2)),
            ]
        )
        self.deconv_list = nn.ModuleList(
            [
                nn.Sequential(nn.BatchNorm2d(decoder_channels[0]), nn.ReLU(), nn.ConvTranspose2d(decoder_channels[0], decoder_channels[1], 2, 2, 0)),
                nn.Sequential(nn.BatchNorm2d(decoder_channels[1]), nn.ReLU(), nn.ConvTranspose2d(decoder_channels[1], decoder_channels[2], 3, 2, 0)),
                nn.Sequential(nn.BatchNorm2d(decoder_channels[2]), nn.ReLU(), nn.ConvTranspose2d(decoder_channels[2], decoder_channels[3], 2, 2, 0)),
                nn.Sequential(nn.BatchNorm2d(decoder_channels[3]), nn.ReLU(), nn.ConvTranspose2d(decoder_channels[3], decoder_channels[4], 2, 2, 0)),
            ]
        )
        self.stem = nn.Sequential(nn.Conv2d(in_channels, encoder_channels[0], 7, 1, 3), nn.BatchNorm2d(encoder_channels[0]), nn.ReLU())
        self.sr_block_1 = SR_Block(encoder_channels[0], encoder_channels[0], False)
        self.sr_block_2 = SR_Block(encoder_channels[0], encoder_channels[1], False)
        self.sr_block_3 = SR_Block(encoder_channels[1], encoder_channels[2], False)
        self.sr_block_4 = SR_Block(encoder_channels[2], encoder_channels[3], False)
        assert encoder_channels[4] == decoder_channels[0], 'encoder_channels[4] != decoder_channels[0].'
        self.sr_block_5 = SR_Block(encoder_channels[3], encoder_channels[4], False)
        self.aspp_1 = ASPP(encoder_channels[4], decoder_channels[0])
        self.pga_1 = PGA(encoder_channels[3], decoder_channels[1])
        self.pga_2 = PGA(encoder_channels[2], decoder_channels[2])
        self.pga_3 = PGA(encoder_channels[1], decoder_channels[3])
        self.pga_4 = PGA(encoder_channels[0], decoder_channels[4])
        self.sr_block_6 = SR_Block((encoder_channels[3] + decoder_channels[1]) // 2, decoder_channels[1])
        self.sr_block_7 = SR_Block((encoder_channels[2] + decoder_channels[2]) // 2, decoder_channels[2])
        self.sr_block_8 = SR_Block((encoder_channels[1] + decoder_channels[3]) // 2, decoder_channels[3])
        self.sr_block_9 = SR_Block((encoder_channels[0] + decoder_channels[4]) // 2, decoder_channels[4])
        self.aspp_2 = ASPP(decoder_channels[4], decoder_channels[-1])
        self.output_layer = nn.Sequential(nn.BatchNorm2d(decoder_channels[-1]), nn.ReLU(), nn.Conv2d(decoder_channels[-1], out_channels, 1, 1, 0))

    def forward(self, x):
        feature_1 = self.sr_block_1(self.stem(x))
        feature_2 = self.sr_block_2(self.pool_list[0](feature_1))
        feature_3 = self.sr_block_3(self.pool_list[1](feature_2))
        feature_4 = self.sr_block_4(self.pool_list[2](feature_3))
        feature_5 = self.sr_block_5(self.pool_list[3](feature_4))
        feature_5 = self.aspp_1(feature_5)
        de_feature_1 = self.sr_block_6(self.pga_1(feature_4, self.deconv_list[0](feature_5)))
        de_feature_2 = self.sr_block_7(self.pga_2(feature_3, self.deconv_list[1](de_feature_1)))
        de_feature_3 = self.sr_block_8(self.pga_3(feature_2, self.deconv_list[2](de_feature_2)))
        de_feature_4 = self.sr_block_9(self.pga_4(feature_1, self.deconv_list[3](de_feature_3)))
        de_feature_4 = self.aspp_2(de_feature_4)
        output = self.output_layer(de_feature_4)
        return output


if __name__ == '__main__':
    config = {
        'size': 180,
        'in_channels': 1,
        'encoder_channels': [32, 64, 128, 256, 512],
        'decoder_channels': [512, 256, 128, 64, 32],
        'out_channels': 2,
    }
    net = PG_SN(config)
    print(net(torch.randn(8, 1, 180, 180)).shape)
    print(torchsummary.summary(net, (1, 180, 180), batch_size=-1, device='cpu'))
