# Importing 
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

def calc_mean_std(features):
    batch_size, c = features.size()[:2]
    features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
    return features_mean, features_std

def adain(content_features, style_features):
    content_mean, content_std = calc_mean_std(content_features)
    style_mean, style_std = calc_mean_std(style_features)
    normalized_features = style_std * (content_features - content_mean) / content_std + style_mean
    return normalized_features

# Layer for Decoder
class RC(nn.Module):
    """A wrapper of ReflectionPad2d and Conv2d"""
    def __init__(self, in_channels, out_channels, kernel_size=3, pad_size=1, activated=True):
        super().__init__()
        self.pad = nn.ReflectionPad2d((pad_size, pad_size, pad_size, pad_size))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.activated = activated

    def forward(self, x):
        h = self.pad(x)
        h = self.conv(h)
        if self.activated:
            return F.relu(h)
        else:
            return h

# Decoder model
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rc1 = RC(512, 256, 3, 1)
        self.rc2 = RC(256, 256, 3, 1)
        self.rc3 = RC(256, 256, 3, 1)
        self.rc4 = RC(256, 256, 3, 1)
        self.rc5 = RC(256, 128, 3, 1)
        self.rc6 = RC(128, 128, 3, 1)
        self.rc7 = RC(128, 64, 3, 1)
        self.rc8 = RC(64, 64, 3, 1)
        self.rc9 = RC(64, 3, 3, 1, False)
    
    def forward(self, features):
        h = self.rc1(features)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc2(h)
        h = self.rc3(h)
        h = self.rc4(h)
        h = self.rc5(h)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc6(h)
        h = self.rc7(h)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc8(h)
        h = self.rc9(h)
        return h

# Encoder Model
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained = True).features
        self.slice1 = vgg[:2]
        self.slice2 = vgg[2:7]
        self.slice3 = vgg[7:12]
        self.slice4 = vgg[12:21]
        for p in self.parameters():
            p.requires_grad = False
    
    def forward(self, images, out_last = False):
        h1 = self.slice1(images)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        if out_last:
            return h4
        else:
            return h1, h2, h3, h4

# Model for both Encoder/Decoder and forward pass
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_encoder  = Encoder()
        self.decoder = Decoder()

    @staticmethod
    def calc_content_loss(out_features, t):
        return F.mse_loss(out_features, t)
  
    @staticmethod
    def calc_style_loss(content_mid, style_mid):
        loss = 0
        for c, s in zip(content_mid, style_mid):
            c_mean, c_std = calc_mean_std(c)
            s_mean, s_std = calc_mean_std(s)
            loss += F.mse_loss(c_mean, s_mean) + F.mse_loss(c_std, s_std)
    
        return loss

    def generate(self, content, style, alpha = 1.0):
        content_features = self.vgg_encoder(content, out_last = True)
        style_features = self.vgg_encoder(style, out_last = True)
        # AdaIN function
        t = adain(content_features, style_features)
        # Alpha change (higher alpha, more like style)
        t = alpha * t + (1 - alpha) * content_features
        out = self.decoder(t)
        return out
  
    def forward(self, content, style, alpha = 1.0, la = 10):
        content_features = self.vgg_encoder(content, out_last = True)
        style_features = self.vgg_encoder(style, out_last = True)
        t = adain(content_features, style_features)
        t = alpha * t + (1 - alpha) * content_features
        out = self.decoder(t)

        out_features = self.vgg_encoder(out, out_last = True)
        out_middle_features = self.vgg_encoder(out, out_last = False)
        style_middle_features = self.vgg_encoder(style, out_last = False)

        loss_c = self.calc_content_loss(out_features, t)
        loss_s = self.calc_style_loss(out_middle_features, style_middle_features)
        loss = loss_c + la *loss_s
        return loss
