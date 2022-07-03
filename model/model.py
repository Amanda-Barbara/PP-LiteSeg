import torch
import torch.nn as nn
import torch.nn.functional as F

from .stdc import STDCNet, ConvX


class SPPM(nn.Module):
    """Simple Pyramid Pooling Module
    """
    def __init__(self, in_channels, inter_channels, out_channels, bin_sizes) -> None:
        """
        :param in_channels: int, channels of input feature
        :param inter_channels: int, chennels of mid conv
        :param out_channels: int, channels of output feature
        :param bin_sizes: list, avg pool size of 3 features
        """
        super().__init__()

        self.stage1_pool = nn.AdaptiveAvgPool2d(output_size=bin_sizes[0])
        self.stage1_conv = ConvX(in_channels, inter_channels, kernel_size=1)

        self.stage2_pool = nn.AdaptiveAvgPool2d(output_size=bin_sizes[1])
        self.stage2_conv = ConvX(in_channels, inter_channels, kernel_size=1)

        self.stage3_pool = nn.AdaptiveAvgPool2d(output_size=bin_sizes[2])
        self.stage3_conv = ConvX(in_channels, inter_channels, kernel_size=1)

        self.conv_out = ConvX(inter_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        h, w =  x.size()[2:]

        f1 = self.stage1_pool(x)
        f1 = self.stage1_conv(f1)
        f1 =  F.interpolate(f1, (h, w), mode='bilinear', align_corners=False)
    
        f2 = self.stage2_pool(x)
        f2 = self.stage2_conv(f2)
        f2 =  F.interpolate(f2, (h, w), mode='bilinear', align_corners=False)

        f3 = self.stage3_pool(x)
        f3 = self.stage3_conv(f3)
        f3 =  F.interpolate(f3, (h, w), mode='bilinear', align_corners=False)

        x = self.conv_out(f1 + f2 + f3)

        return x


class UAFM(nn.Module):
    """Unified Attention Fusion Modul
    """
    def __init__(self, low_chan, hight_chan, out_chan, u_type='sp') -> None:
        """
        :param low_chan: int, channels of input low-level feature
        :param hight_chan: int, channels of input high-level feature
        :param out_chan: int, channels of output faeture
        :param u_type: string, attention type, sp: spatial attention, ch: channel attention
        """
        super().__init__()
        self.u_type = u_type

        if u_type == 'sp':
            self.conv_atten = nn.Sequential(
                ConvX(4, 2, kernel_size=3),
                nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(1),
                )
        else:
            self.conv_atten = nn.Sequential(
                ConvX(4 * hight_chan, hight_chan // 2,  kernel_size=1, bias=False, act="leaky"),
                nn.Conv2d(hight_chan // 2, hight_chan, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(hight_chan),
            )

        self.conv_low = ConvX(low_chan, hight_chan, kernel_size=3, padding=1, bias=False)
        self.conv_out = ConvX(hight_chan, out_chan, kernel_size=3, padding=1, bias=False)

    def _spatial_attention(self, x):
        """
        :param x: tensor, feature
        :return x: tensor, fused feature
        """
        mean_value = torch.max(x, dim=1, keepdim=True)[0]
        max_value = torch.mean(x, dim=1, keepdim=True)

        value = torch.concat([mean_value, max_value], dim=1)

        return value

    def _channel_attention(self, x):
        """
        :param x: tensor, feature
        :return x: tensor, fused feature
        """
        avg_value = F.adaptive_avg_pool2d(x, 1)
        max_value = torch.max(torch.max(x, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
        value = torch.concat([avg_value, max_value], dim=1)

        return value


    def forward(self, x_high, x_low):
        """
        :param x_high: tensor, high-level feature
        :param x_low: tensor, low-level feature
        :return x: tensor, fused feature
        """
        h, w =  x_low.size()[2:]

        x_low = self.conv_low(x_low)
        x_high = F.interpolate(x_high, (h, w), mode='bilinear', align_corners=False)

        if self.u_type == 'sp':
            atten_high = self._spatial_attention(x_high)
            atten_low = self._spatial_attention(x_low)
        else:
            atten_high = self._channel_attention(x_high)
            atten_low = self._channel_attention(x_low)

        atten = torch.concat([atten_high, atten_low], dim=1)
        atten = torch.sigmoid(self.conv_atten(atten))

        x = x_high * atten + x_low * (1 - atten)
        x = self.conv_out(x)

        return x


class SegHead(nn.Module):
    """FLD Decoder
    """
    def __init__(self, bin_sizes, decode_chans):
        """
        :param bin_sizes: list, avg pool size of 3 features
        :param decode_chans: list, channels of decoder feature size
        """
        super().__init__()

        self.sppm = SPPM(1024, decode_chans[0], decode_chans[0], bin_sizes)
        self.uafm1 = UAFM(512, decode_chans[0], decode_chans[1])
        self.uafm2 = UAFM(256, decode_chans[1], decode_chans[2])

    def forward(self, x):
        # x8, x16, x32
        sppm_feat = self.sppm(x[-1])
  
        merge_feat1 = self.uafm1(sppm_feat, x[1])
        merge_feat2 = self.uafm2(merge_feat1, x[0])

  
        return [sppm_feat, merge_feat1, merge_feat2]


class SegClassifier(nn.Module):
    """Classification Layer
    """
    def __init__(self, in_chan, mid_chan, n_classes) -> None:
        """
        :param in_chan: int, channels of input feature
        :param mid_chan: int, channels of mid conv
        :param n_classes: int, number of classification
        """
        super().__init__()
        self.conv = ConvX(in_chan, mid_chan, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)

        return x


class PPLiteSeg(nn.Module):
    def __init__(self, n_classes, t='stdc1') -> None:
        """
        :param n_classes: int, number of classification
        :param t: string, backbone type, stdc1/stdc2
        """
        super().__init__()

        if t == 'stdc1':
            layers=[2, 2, 2]
        else:
            layers=[4, 5, 3]
        backbone_indices=[2, 3, 4]
        bin_sizes=[1, 2, 4]
        decode_chans = [128, 96, 64]
  
        self.backbone = STDCNet(64, layers)
        self.backbone_indices = backbone_indices
        self.seg_head = SegHead(bin_sizes, decode_chans)

        self.classifer = []
        for chan in decode_chans:
            cls = SegClassifier(chan, 64, n_classes)
            self.classifer.append(cls)

    def forward(self, x):
        h, w = x.size()[2:]

        feats_backbone = self.backbone(x)  # [x2, x4, x8, x16, x32]
        feats_selected = [feats_backbone[i] for i in self.backbone_indices]

        head_out = self.seg_head(feats_selected)

        if self.training:
            outs = []
            for i in range(3):
                x = F.interpolate(head_out[i], (h, w), mode='bilinear', align_corners=False)
                x = self.classifer[i](x)
                outs.append(x)
        else:
            outs = F.interpolate(head_out[-1], (h, w), mode='bilinear', align_corners=False)
            outs = self.classifer[-1](outs)
            outs = torch.softmax(outs, dim=1)

        return outs


if __name__ == '__main__':
    model = PPLiteSeg(4)
    model.eval()
    model.to('cpu')
    x = torch.zeros((1, 3, 512, 1024))
    y = model(x)
    print(y.size())
