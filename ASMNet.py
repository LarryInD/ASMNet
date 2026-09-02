import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from torchvision.models import resnet50
import numpy
import math
from mmrotate.models.builder import ROTATED_BACKBONES
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath
from timm.models.registry import register_model
# from mmdet.utils import get_root_logger
from mmrotate.utils import get_root_logger
from mmcv.runner import _load_checkpoint



def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'ASMNet': _cfg(crop_pct=0.9, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
}


class Stem(nn.Module):
    def __init__(self, input_dim, output_dim, activation=nn.GELU):
        super(Stem, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, output_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim // 2),
            nn.GELU(),
            nn.Conv2d(output_dim // 2, output_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.GELU()   
        )
        
    def forward(self, x):
        return self.stem(x)

    
class MLP(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, drop=0., mid_conv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mid_conv = mid_conv
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

        if self.mid_conv:
            self.mid = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features)
            self.mid_norm = nn.BatchNorm2d(hidden_features)

        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.norm2 = nn.BatchNorm2d(out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)

        if self.mid_conv:
            x_mid = self.mid(x)
            x_mid = self.mid_norm(x_mid)
            x = self.act(x_mid)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.drop(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, mid_conv=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))
        else:
            x = x + self.drop_path(self.mlp(x))
        return x



class LiteAxisRCMContrast(nn.Module):
    """
    Lightweight Axis-Decoupled Target-Aware Sparse Mining.

    Args:
        in_channels: input channels
        out_channels: output channels
        K: base sparse offset
        axis: 'h' for height-wise mining, 'w' for width-wise mining
        num_prototypes: number of sparse offsets, default 4
        hop_factors: offset factors, default (1, 2), giving {K, 2K, -K, -2K}
        reduction: hidden channel reduction for lightweight 1x1 projections
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        K=2,
        axis="h",
        num_prototypes=4,
        hop_factors=(1, 2),
        reduction=4,
    ):
        super().__init__()

        if axis not in ("h", "w"):
            raise ValueError(f"axis must be 'h' or 'w', but got {axis}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.axis = axis

        # Sparse offsets: {K, 2K, -K, -2K} when hop_factors=(1, 2)
        self.shifts = [K * f for f in hop_factors] + [-K * f for f in hop_factors]
        self.num_offsets = len(self.shifts)

        if num_prototypes is not None and num_prototypes != self.num_offsets:
            raise ValueError(
                f"num_prototypes={num_prototypes} does not match "
                f"the number of offsets {self.num_offsets}. "
                f"Please set num_prototypes={self.num_offsets} or adjust hop_factors."
            )

        hidden_channels = max(8, in_channels // reduction)

        self.axis_target = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.offset_selector = nn.Sequential(
            nn.Conv2d(
                in_channels * self.num_offsets,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                self.num_offsets,
                kernel_size=1,
                bias=True,
            ),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def _axis_pool(self, x):
        """
        Coordinate-preserving directional pooling.

        axis='h': aggregate along width, preserve height -> B,C,H,1
        axis='w': aggregate along height, preserve width -> B,C,1,W
        """
        if self.axis == "h":
            return x.mean(dim=3, keepdim=True)
        return x.mean(dim=2, keepdim=True)

    def _roll(self, x, shift):
        """
        Axis-wise sparse shift.
        """
        if self.axis == "h":
            return torch.roll(x, shifts=shift, dims=2)
        return torch.roll(x, shifts=shift, dims=3)

    def forward(self, x):
        """
        Args:
            x: input feature, shape B,C,H,W

        Returns:
            output feature, shape B,out_channels,H,W
        """
        axis_desc = self._axis_pool(x)
        axis_att = self.axis_target(axis_desc)

        modulated_diffs = []
        relation_descs = []

        for shift in self.shifts:
            x_rolled = self._roll(x, shift)
            diff = x_rolled - x
            diff_t = diff * axis_att
            modulated_diffs.append(diff_t)
            relation_descs.append(self._axis_pool(diff_t))

        relation_desc = torch.cat(relation_descs, dim=1)
        logits = self.offset_selector(relation_desc)
        alpha = F.softmax(logits, dim=1)
        x_sparse = 0.0

        for idx, diff_t in enumerate(modulated_diffs):
            weight = alpha[:, idx:idx + 1, :, :]
            x_sparse = x_sparse + weight * diff_t
        out = torch.cat([x, x_sparse], dim=1)
        out = self.fuse(out)

        return out


class ASM_Height(LiteAxisRCMContrast):
    def __init__(
        self,
        in_channels,
        out_channels,
        K=2,
        num_prototypes=4,
        hop_factors=(1, 2),
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            K=K,
            axis="h",
            num_prototypes=num_prototypes,
            hop_factors=hop_factors,
        )


class ASM_Width(LiteAxisRCMContrast):
    def __init__(
        self,
        in_channels,
        out_channels,
        K=2,
        num_prototypes=4,
        hop_factors=(1, 2),
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            K=K,
            axis="w",
            num_prototypes=num_prototypes,
            hop_factors=hop_factors,
        )


class ASM(nn.Module):
    """
    ASM module
    """
    def __init__(self, in_channels, drop_path=0.0, K=2):
        super(ASM, self).__init__()
        self.channels = in_channels
        self.K = K
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv_h = ASM_Height(in_channels, in_channels * 2, K=self.K)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )  
        self.graph_conv_w = ASM_Width(in_channels, in_channels * 2, K=self.K)
        self.fc3 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )  
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        x = self.graph_conv_h(x)
        x = self.fc2(x)
        x = self.graph_conv_w(x)
        x = self.fc3(x)
        x = self.drop_path(x) + _tmp
        return x


class Downsample(nn.Module):
    """ Convolution-based downsample
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()        
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features # same as input
        hidden_features = hidden_features or in_features # x4
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = nn.GELU()
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x


class ASMNet(torch.nn.Module):
    def __init__(self, local_blocks, local_channels,
                 global_blocks, global_channels,
                 dropout=0., drop_path=0., emb_dims=512,
                 K=2, distillation=True, num_classes=1000,
                 pretrained=None, 
                 
                 out_indices=None):
        super(ASMNet, self).__init__()

        self.distillation = distillation
        self.out_indices = out_indices
        self.pretrained = pretrained
        n_blocks = sum(global_blocks) + sum(local_blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path, n_blocks)]  
        dpr_idx = 0
        self.stem = Stem(input_dim=3, output_dim=local_channels[0])
    
        self.local_backbone = nn.ModuleList([])
        for i in range(len(local_blocks)):
            if i > 0:
                self.local_backbone.append(Downsample(local_channels[i-1], local_channels[i]))
            for _ in range(local_blocks[i]):
                self.local_backbone.append(InvertedResidual(dim=local_channels[i], mlp_ratio=4, drop_path=dpr[dpr_idx]))
                dpr_idx += 1
        self.local_backbone.append(Downsample(local_channels[-1], global_channels[0]))  

        self.backbone = nn.ModuleList([])
        for i in range(len(global_blocks)):
            if i > 0:
                self.backbone.append(Downsample(global_channels[i-1], global_channels[i]))
            for j in range(global_blocks[i]):
                self.backbone += [nn.Sequential(
                                    ASM(global_channels[i], drop_path=dpr[dpr_idx], K=K),
                                    FFN(global_channels[i], global_channels[i] * 4, drop_path=dpr[dpr_idx]),
                                    )
                                    ]
                dpr_idx += 1

        self.init_weights()
        self = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
        

    def init_weights(self):
        logger = get_root_logger()
        if  self.pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            print("Pretrained weights being loaded")
            logger.warn('Pretrained weights being loaded')
            ckpt_path = self.pretrained
            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            # print("ckpt keys: ", ckpt.keys())
            if 'state_dict' in ckpt:
                
                # _state_dict = ckpt['state_dict_ema']
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            
            
            new_state_dict = {}
            
            for key, value in _state_dict.items():
                if key.startswith('backbone.'):
                    new_key = key[len('backbone.'):]  
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value 

            _state_dict = new_state_dict

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)
            logger.info(f"Miss {missing_keys}")
            logger.info(f"Unexpected {unexpected_keys}")
            
    
    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        x = self.stem(inputs)
        outs = []
        for i in range(len(self.local_backbone)):
            x = self.local_backbone[i](x)
            if i in self.out_indices:
                outs.append(x)
        offset = len(self.local_backbone)
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
            idx = i + offset
            if idx in self.out_indices:
                outs.append(x)

        return outs



@ROTATED_BACKBONES.register_module()
def ASMNet_T_ESWA(pretrained=False, **kwargs):
    model = ASMNet(local_blocks=[3, 3, 9],
                      local_channels=[42, 84, 176],
                      global_blocks=[3],
                      global_channels=[256],
                      dropout=0.,
                      drop_path=0.1,
                      emb_dims=512,
                      K=2,
                      distillation=False,
                      num_classes=1000,
                      out_indices=[2, 6, 16, 20],
                    pretrained=None,
                      )
    model.default_cfg = default_cfgs['ASMNet']
    return model

@ROTATED_BACKBONES.register_module()
def ASMNet_T_ESWA_NoNaN(pretrained=False, **kwargs):
    model = ASMNet(local_blocks=[3, 3, 9],
                      local_channels=[42, 84, 176],
                      global_blocks=[2],
                      global_channels=[256],
                      dropout=0.,
                      drop_path=0.1,
                      emb_dims=512,
                      K=2,
                      distillation=False,
                      num_classes=1000,
                      out_indices=[2, 6, 16, 19],
                    pretrained=None,
                      )
    model.default_cfg = default_cfgs['ASMNet']
    return model


@ROTATED_BACKBONES.register_module()
def ASMNet_S_ESWA(pretrained=False, **kwargs):
    model = ASMNet(local_blocks=[2, 2, 4],
                    local_channels=[64, 128, 320],
                    global_blocks=[2],
                    global_channels=[512],
                    dropout=0.,
                    drop_path=0.1,
                    emb_dims=768,
                    K=2,
                    distillation=False,
                    num_classes=1000,
                    out_indices=[1, 4, 9, 12],
                    pretrained=None,
                    )
    model.default_cfg = default_cfgs['ASMNet']
    return model