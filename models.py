import numpy as np
import os
import torch
import cv2
from torch import nn
from modules.blocks import LinearBlock, Conv2dBlock, ResBlocks, ActFirstResBlock
from modules.VGG import vgg19_bn


class GenModel_FC(nn.Module):
    def __init__(self, text_max_len):
        super(GenModel_FC, self).__init__()
        self.enc_image = ImageEncoder()
        self.enc_text = TextEncoder_FC(text_max_len)
        self.dec = Decoder()
        self.linear_mix = nn.Linear(1024, 512)

    def decode(self, content, adain_params):
        # decode content and style codes to an image
        assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    # feat_mix: b,1024,8,27
    def mix(self, feat_xs, feat_embed):
        feat_mix = torch.cat([feat_xs, feat_embed], dim=1) # b,1024,8,27
        f = feat_mix.permute(0, 2, 3, 1)
        ff = self.linear_mix(f) # b,8,27,1024->b,8,27,512
        return ff.permute(0, 3, 1, 2)


class TextEncoder_FC(nn.Module):
    def __init__(self, text_max_len):
        super(TextEncoder_FC, self).__init__()
        embed_size = 64
        vocab_size = 55
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.fc = nn.Sequential(
                nn.Linear(text_max_len*embed_size, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=False),
                nn.Linear(1024, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=False),
                nn.Linear(2048, 4096)
                )
        '''embed content force'''
        self.linear = nn.Linear(embed_size, 512)

    def forward(self, x, f_xs_shape):
        xx = self.embed(x) # b,t,embed

        batch_size = xx.shape[0]
        xxx = xx.reshape(batch_size, -1) # b,t*embed
        out = self.fc(xxx)

        '''embed content force'''
        xx_new = self.linear(xx) # b, text_max_len, 512
        ts = xx_new.shape[1]
        height_reps = f_xs_shape[-2]
        width_reps = f_xs_shape[-1] // ts
        tensor_list = list()
        for i in range(ts):
            text = [xx_new[:, i:i + 1]] # b, text_max_len, 512
            tmp = torch.cat(text * width_reps, dim=1)
            tensor_list.append(tmp)

        padding_reps = f_xs_shape[-1] % ts
        if padding_reps:
            embedded_padding_char = self.embed(torch.full((1, 1), tokens['PAD_TOKEN'], dtype=torch.long).cuda())
            embedded_padding_char = self.linear(embedded_padding_char)
            padding = embedded_padding_char.repeat(batch_size, padding_reps, 1)
            tensor_list.append(padding)

        res = torch.cat(tensor_list, dim=1) # b, text_max_len * width_reps + padding_reps, 512
        res = res.permute(0, 2, 1).unsqueeze(2) # b, 512, 1, text_max_len * width_reps + padding_reps
        final_res = torch.cat([res] * height_reps, dim=2)

        return out, final_res


'''VGG19_IN tro'''
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.model = vgg19_bn(False)
        self.output_dim = 512

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, ups=3, n_res=2, dim=512, out_dim=1, res_norm='adain', activ='relu', pad_type='reflect'):
        super(Decoder, self).__init__()

        self.model = []
        self.model += [ResBlocks(n_res, dim, res_norm,
                                activation=activ, pad_type=pad_type)]
        for i in range(ups):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2,
                                       norm='in',
                                       activation=activ,
                                       pad_type=pad_type)]
            dim //= 2
        self.model += [Conv2dBlock(dim, out_dim, 7, 1, 3,
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)
