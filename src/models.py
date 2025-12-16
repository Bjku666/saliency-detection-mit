"""模型构建模块

基于 segmentation_models_pytorch 按 backbone 名称
构建用于显著性预测的 U-Net 分割模型。
"""

import segmentation_models_pytorch as smp
import torch

def build_model(backbone_name):
    """
    方案 A: backbone_name = 'efficientnet-b7'
    方案 B: backbone_name = 'mit_b5' (SegFormer Transformer)
    """
    print(f"Building Model with backbone: {backbone_name}")
    
    # SMP 库会自动处理 CNN 和 Transformer 的差异
    # 使用标准 Unet，保持 mit_b5 Transformer 编码器的全局感受野
    model = smp.Unet(
        encoder_name=backbone_name, 
        encoder_weights="imagenet",     
        in_channels=3,                  
        classes=1,                      
        activation=None
    )
    
    return model