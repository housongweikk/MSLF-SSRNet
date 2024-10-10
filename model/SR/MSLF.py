import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
from thop import profile

class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        channels = 64
        n_block = 16
        self.angRes = args.angRes_in
        self.factor = args.scale_factor
        self.init_conv = nn.Conv2d(1, channels, kernel_size=3, stride=1, dilation=self.angRes, padding=self.angRes,
                                   bias=False)
        self.init_conv1 = CustomConvBlockSconv(1, channels, kernel_size=3)
        self.extract = tsfefbs(n_block, self.angRes, channels)
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * self.factor ** 2, kernel_size=1, stride=1, padding=0),
            nn.PixelShuffle(self.factor),
            nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x, info=None):
        x_upscale = F.interpolate(x, scale_factor=self.factor, mode='bilinear', align_corners=False)
        x1 = SAI2MacPI(x, self.angRes)
        buffer = self.init_conv(x1)
        buffer1= self.init_conv1(x)
        buffer2 = self.extract(buffer,buffer1)
        buffer_SAI = MacPI2SAI(buffer2, self.angRes)
        out = self.upsample(buffer_SAI) + x_upscale
        return out


class tsfefbs(nn.Module):
    def __init__(self, n_block, angRes, channels):
        super(tsfefbs, self).__init__()
        self.n_block = n_block
        self.angRes = angRes
        Blocks = [tsfefb(angRes, channels) for i in range(n_block)]
        self.Blocks = nn.ModuleList(Blocks)


    def forward(self,x,x1):
        buffer=x
        buffer1=x1
        for block in self.Blocks:
            buffer= block(buffer, buffer1)
            buffer1=MacPI2SAI(buffer, self.angRes)
        return buffer

class tsfefb(nn.Module):
    def __init__(self, angRes, channels):
        super(tsfefb, self).__init__()
        self.angRes = angRes
        self.channels = channels
        self.SpaConv = nn.Sequential(
            CustomConvBlockS(channels, channels, kernel_size=3),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.AngConv = nn.Sequential(
            CustomConvBlockA(channels, channels, kernel_size=3),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.EPIConv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=[1, 3*angRes], stride=[1, angRes], padding=[0, angRes], bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, angRes * channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            PixelShuffle1D(angRes),
        )
        self.attention_fusion = AttentionFusion(channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(4 * channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),bias=False),
        )

    def forward(self, x, x1):
        feaSpa = self.SpaConv(x1)
        feaAng = self.AngConv(x)
        feaEpiH = self.EPIConv(x)
        feaEpiV = self.EPIConv(x.permute(0, 1, 3, 2).contiguous()).permute(0, 1, 3, 2)
        buffer = torch.cat((feaSpa, feaAng, feaEpiH, feaEpiV), dim=1)
        [out, att_weight] = self.attention_fusion(buffer)
        buffer = self.fuse(out)
        return buffer + x


class AttentionFusion(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super(AttentionFusion, self).__init__()
        self.epsilon = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, N, C, height, width = x.size()
        x_reshape = x.view(m_batchsize, N, -1)
        M = C * height * width

        # compute covariance feature
        mean = torch.mean(x_reshape, dim=-1).unsqueeze(-1)
        x_reshape = x_reshape - mean
        cov = (1 / (M - 1) * x_reshape @ x_reshape.transpose(-1, -2)) * self.alpha
        norm = cov / ((cov.pow(2).mean((1, 2), keepdim=True) + self.epsilon).pow(0.5))

        expanded_norm = torch.cat((self.gamma * norm + self.beta, self.gamma * norm + self.beta), dim=-2)
        attention = F.glu(expanded_norm, dim=-2)

        x_reshape = x.view(m_batchsize, N, -1)

        out = torch.bmm(attention, x_reshape)
        out = out.view(m_batchsize, N, C, height, width)
        out += x
        out = out.view(m_batchsize, -1, height, width)

        return out, attention



class PixelShuffle1D(nn.Module):
    """
    1D pixel shuffler
    Upscales the last dimension (i.e., W) of a tensor by reducing its channel length
    inout: x of size [b, factor*c, h, w]
    output: y of size [b, c, h, w*factor]
    """

    def __init__(self, factor):
        super(PixelShuffle1D, self).__init__()
        self.factor = factor

    def forward(self, x):
        b, fc, h, w = x.shape
        c = fc // self.factor
        x = x.contiguous().view(b, self.factor, c, h, w)
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # b, c, h, w, factor
        y = x.view(b, c, h, w * self.factor)
        return y





class CustomConvBlockA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(CustomConvBlockA, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
            bias=False
        )

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        unfold = nn.Unfold(kernel_size=5, stride=5)
        x_unfolded = unfold(x)

        x_unfolded = x_unfolded.transpose(1, 2)
        x_unfolded = x_unfolded.contiguous().view(-1, channels, 5, 5)
        output_blocks = self.conv(x_unfolded)

        output_blocks = output_blocks.view(batch_size, -1, output_blocks.shape[1] * 5 * 5)
        output_blocks = output_blocks.transpose(1, 2)

        fold = nn.Fold(output_size=(height, width), kernel_size=5, stride=5)
        output = fold(output_blocks)

        return output





class CustomConvBlockS(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(CustomConvBlockS, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
            bias=False
        )

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        assert height % 5 == 0 and width % 5 == 0, "Height and Width must be divisible by 5"

        small_h = height // 5
        small_w = width // 5

        unfold = nn.Unfold(kernel_size=(small_h, small_w), stride=(small_h, small_w))
        x_unfolded = unfold(x)  # [batch_size, channels*small_h*small_w, 25]

        x_unfolded = x_unfolded.view(batch_size, channels, small_h, small_w, 5, 5)
        x_unfolded = x_unfolded.permute(0, 4, 5, 1, 2, 3).contiguous().view(-1, channels, small_h, small_w)

        output_blocks = self.conv(x_unfolded)

        output_blocks = output_blocks.view(batch_size, 5, 5, channels, small_h, small_w)
        output_blocks = output_blocks.permute(0, 3, 4, 1, 5, 2).contiguous().view(batch_size, channels, height, width)

        return output_blocks



class CustomConvBlockSconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, num_blocks=5):
        super(CustomConvBlockSconv, self).__init__()
        self.num_blocks = num_blocks
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
            bias=False
        )

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        assert height % self.num_blocks == 0 and width % self.num_blocks == 0, "Height and Width must be divisible by the number of blocks"

        small_h = height // self.num_blocks
        small_w = width // self.num_blocks

        unfold = nn.Unfold(kernel_size=(small_h, small_w), stride=(small_h, small_w))
        x_unfolded = unfold(x)  # [batch_size, channels*small_h*small_w, 25]

        x_unfolded = x_unfolded.view(batch_size, -1, small_h, small_w)
        x_unfolded = x_unfolded.transpose(1, 2).contiguous().view(-1, channels, small_h, small_w)

        output_blocks = self.conv(x_unfolded)

        output_blocks = output_blocks.view(batch_size, self.num_blocks, self.num_blocks, -1, small_h, small_w)
        output_blocks = output_blocks.permute(0, 3, 1, 4, 2, 5).contiguous()
        output_blocks = output_blocks.view(batch_size, -1, height, width)

        return output_blocks



def MacPI2SAI(x, angRes):
    b, c, hu, wv = x.shape
    h, w = hu // angRes, wv // angRes
    x_reshaped = x.view(b, c, angRes, h, angRes, w)
    x_reshaped = x_reshaped.permute(0, 1, 3, 5, 2, 4).contiguous()
    x_reshaped = x_reshaped.view(b, c, h * angRes, w * angRes)
    return x_reshaped
    
def SAI2MacPI(x, angRes):
    b, c, hu, wv = x.shape
    h, w = hu // angRes, wv // angRes
    tempU = []
    for i in range(h):
        tempV = []
        for j in range(w):
            tempV.append(x[:, :, i::h, j::w])
        tempU.append(torch.cat(tempV, dim=3))
    out = torch.cat(tempU, dim=2)
    return out




class get_loss(nn.Module):
    def __init__(self, alpha=0.7, edge_threshold=0.1):
        super(get_loss, self).__init__()
        self.alpha = alpha
        self.edge_threshold = edge_threshold

    def sobel_edges(self, x):
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).reshape(
            (1, 1, 3, 3)).to(x.device)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).reshape(
            (1, 1, 3, 3)).to(x.device)
        edge_x = F.conv2d(x, sobel_x, padding=1)
        edge_y = F.conv2d(x, sobel_y, padding=1)
        edges = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
        edges = edges / edges.max()

        binary_edges = (edges > self.edge_threshold).float()
        return binary_edges

    def forward(self, sr, hr, criterion_data=[]):
        edges = self.sobel_edges(hr)
        pixel_mae = torch.abs(sr - hr)
        weighted_mae = edges * pixel_mae
        total_pixels = hr.numel()
        edge_weighted_mae = weighted_mae.sum() / total_pixels
        total_mae = pixel_mae.mean()
        combined_loss = self.alpha * total_mae + (1 - self.alpha) * edge_weighted_mae
        return combined_loss


def weights_init(m):
    pass
# if __name__ == '__main__':
#     class Config:
#         channels = 64
#         angRes_in = 5
#         scale_factor = 4
#
#     args = Config()
#
#     model = get_model(args)
#
#     print(model)
if __name__ == "__main__":
    args = argparse.Namespace(angRes_in=5, scale_factor=4)
    net = get_model(args).cuda()
    input = torch.randn(1, 1, 32 * args.angRes_in, 32 * args.angRes_in).cuda()
    flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.2fM' % (params / 1e6))
    print('   Number of FLOPs: %.2fG' % (flops / 1e9))
