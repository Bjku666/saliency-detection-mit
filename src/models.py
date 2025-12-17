# src/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

# --- TranSalNet 组件 ---

class TransformerEncoder(nn.Module):
    def __init__(self, d_model=768, nhead=8, num_layers=2, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 可学习的位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, d_model, 64, 64) * 0.02) 

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        # 插值调整位置编码以适应当前分辨率 (虽说是固定分辨率，但为了鲁棒性)
        pos = F.interpolate(self.pos_embedding, size=(H, W), mode='bilinear', align_corners=True)
        
        # [B, C, H, W] -> [B, C, H*W] -> [H*W, B, C]
        x_flat = (x + pos).flatten(2).permute(2, 0, 1)
        
        out = self.transformer_encoder(x_flat)
        
        # [H*W, B, C] -> [B, C, H, W]
        out = out.permute(1, 2, 0).view(B, C, H, W)
        return out

class TranSalNet(nn.Module):
    def __init__(self, load_salicon=False):
        super(TranSalNet, self).__init__()
        
        # 1. 骨干网络 (ResNet50)
        # 如果还要加载 SALICON 权重，这里 pretrained=True 也没关系，会被覆盖
        resnet = resnet50(pretrained=True) 
        
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1 # 256, H/4
        self.layer2 = resnet.layer2 # 512, H/8
        self.layer3 = resnet.layer3 # 1024, H/16
        self.layer4 = resnet.layer4 # 2048, H/32

        # 2. 通道调整 (为了进入 Transformer)
        # TranSalNet 官方通常将不同层特征统一投影到 768 维度
        self.conv_process1 = nn.Conv2d(2048, 768, 1) # for layer4
        self.conv_process2 = nn.Conv2d(1024, 768, 1) # for layer3
        self.conv_process3 = nn.Conv2d(512, 768, 1)  # for layer2

        # 3. Transformer Encoders (核心)
        # 官方结构有3个 Transformer 模块
        self.transformer1 = TransformerEncoder(d_model=768, nhead=12, num_layers=2)
        self.transformer2 = TransformerEncoder(d_model=768, nhead=12, num_layers=2)
        self.transformer3 = TransformerEncoder(d_model=768, nhead=8,  num_layers=2)

        # 4. 解码器 / 融合层
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fusion1 = nn.Conv2d(768*2, 768, 3, padding=1)
        self.fusion2 = nn.Conv2d(768*2, 384, 3, padding=1)
        
        self.decoder_final = nn.Sequential(
            nn.Conv2d(384, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True), # 回到 H/4
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True), # 回到 H
            nn.Conv2d(64, 1, 3, padding=1) # 输出 Logits，不带 Sigmoid
        )
        
        # 5. 加载 "核武器" (SALICON Pretrained Weights)
        if load_salicon:
            self.load_pretrained_weights()

    def load_pretrained_weights(self):
        # 假设权重放在 ./pretrained/ 目录下
        path = './pretrained/transalnet_salicon.pth'
        try:
            print(f"Loading SALICON weights from {path} ...")
            # 必须用 strict=False，因为官方权重可能包含多余的训练参数
            # 且我们的 decoder_final 可能与官方不同（为了适配你的 Loss 移除了 Sigmoid）
            checkpoint = torch.load(path, map_location='cpu')
            
            # 处理可能的字典嵌套
            state_dict = checkpoint.get('state_dict', checkpoint)
            if 'model' in state_dict: state_dict = state_dict['model']

            # 过滤掉形状不匹配的层 (比如最后的输出层)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            
            msg = self.load_state_dict(pretrained_dict, strict=False)
            print(f" SALICON Weights Loaded! Missing keys: {len(msg.missing_keys)}")
        except Exception as e:
            print(f" Failed to load SALICON weights: {e}")
            print(" Running with ImageNet initialization only.")

    def forward(self, x):
        # x: [B, 3, 512, 512]
        
        # Backbone extraction
        x0 = self.layer0(x)
        x1 = self.layer1(x0) 
        x2 = self.layer2(x1) # [B, 512, 64, 64]
        x3 = self.layer3(x2) # [B, 1024, 32, 32]
        x4 = self.layer4(x3) # [B, 2048, 16, 16]

        # Projection
        feat4 = self.conv_process1(x4)
        feat3 = self.conv_process2(x3)
        feat2 = self.conv_process3(x2)

        # Transformer Stage
        # T1 处理最深层
        t1 = self.transformer1(feat4) # [B, 768, 16, 16]
        
        # T2 融合层 (上采样 T1 + feat3)
        t1_up = self.up_sample(t1) # 32x32
        # 注意：这里需要处理尺寸可能不匹配的问题 (如果输入不是 32倍数)
        if t1_up.shape != feat3.shape:
             t1_up = F.interpolate(t1, size=feat3.shape[2:], mode='bilinear', align_corners=True)
        
        t2_in = torch.cat([t1_up, feat3], dim=1)
        t2_fuse = self.fusion1(t2_in)
        t2 = self.transformer2(t2_fuse) # [B, 768, 32, 32]

        # T3 融合层 (上采样 T2 + feat2)
        t2_up = self.up_sample(t2) # 64x64
        if t2_up.shape != feat2.shape:
             t2_up = F.interpolate(t2, size=feat2.shape[2:], mode='bilinear', align_corners=True)

        t3_in = torch.cat([t2_up, feat2], dim=1)
        t3_fuse = self.fusion2(t3_in) # -> 384 channels
        t3 = self.transformer3(t3_fuse) # [B, 384, 64, 64] 注意这里维数变了

        # Final Decoder
        out = self.decoder_final(t3) # -> [B, 1, 512, 512]
        
        # 确保输出尺寸和输入一致
        if out.shape[2:] != x.shape[2:]:
             out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)

        return out


# --- 原有工厂函数 ---
import segmentation_models_pytorch as smp

def build_model(backbone_name):
    """
    现在支持: 'mit_b5', 'transalnet'
    """
    print(f"Building Model with backbone: {backbone_name}")
    
    if backbone_name == "transalnet":
        # 实例化并尝试加载 SALICON 权重
        return TranSalNet(load_salicon=True)
    
    else:
        # 旧的 SMP 逻辑
        return smp.Unet(
            encoder_name=backbone_name, 
            encoder_weights="imagenet",     
            in_channels=3,                  
            classes=1,                      
            activation=None
        )