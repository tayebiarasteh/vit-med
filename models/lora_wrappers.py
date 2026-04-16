"""
Created on April 15, 2026.
lora_wrappers.py

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh/
"""


import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from transformers import AutoModel
from huggingface_hub import login
from peft import LoraConfig, TaskType, get_peft_model
import timm





class BackboneWithHead(nn.Module):
    def __init__(self, backbone, head_in_features, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(head_in_features, num_classes)

    def forward(self, x):
        return self.backbone(pixel_values=x)




def _infer_lora_target_modules(backbone, model_family):
    """
    Infer sensible LoRA target module leaf names.

    ViT:
        prefers query/value or q_proj/v_proj or qkv

    ConvNeXt:
        prefers pointwise_conv1/pointwise_conv2 or pwconv1/pwconv2 or fc1/fc2
    """
    linear_leaf_names = sorted({
        name.split('.')[-1]
        for name, module in backbone.named_modules()
        if isinstance(module, nn.Linear)
    })

    conv2d_leaf_names = sorted({
        name.split('.')[-1]
        for name, module in backbone.named_modules()
        if isinstance(module, nn.Conv2d)
    })

    if model_family == 'vit':
        if {'query', 'value'}.issubset(set(linear_leaf_names)):
            targets = ['query', 'value']
        elif {'q_proj', 'v_proj'}.issubset(set(linear_leaf_names)):
            targets = ['q_proj', 'v_proj']
        elif 'qkv' in linear_leaf_names:
            targets = ['qkv']
        else:
            raise RuntimeError(
                f"Could not infer ViT LoRA target modules.\n"
                f"Linear leaf names found: {linear_leaf_names}"
            )

    elif model_family == 'convnext':
        if {'pointwise_conv1', 'pointwise_conv2'}.issubset(set(linear_leaf_names)):
            targets = ['pointwise_conv1', 'pointwise_conv2']
        elif {'pwconv1', 'pwconv2'}.issubset(set(linear_leaf_names)):
            targets = ['pwconv1', 'pwconv2']
        elif {'fc1', 'fc2'}.issubset(set(linear_leaf_names)):
            targets = ['fc1', 'fc2']
        elif {'pointwise_conv1', 'pointwise_conv2'}.issubset(set(conv2d_leaf_names)):
            targets = ['pointwise_conv1', 'pointwise_conv2']
        elif {'pwconv1', 'pwconv2'}.issubset(set(conv2d_leaf_names)):
            targets = ['pwconv1', 'pwconv2']
        else:
            raise RuntimeError(
                f"Could not infer ConvNeXt LoRA target modules.\n"
                f"Linear leaf names found: {linear_leaf_names}\n"
                f"Conv2d leaf names found: {conv2d_leaf_names}"
            )
    else:
        raise ValueError(f"Unsupported model_family: {model_family}")

    print(f"[LoRA] model_family={model_family} | target_modules={targets}")
    return targets




def _build_lora_model(model_name, num_classes, image_size=512, lora_r=16, lora_alpha=16, lora_dropout=0.05):
    """
    Supported LoRA model_name values:
        - vitb_dinov2_lora
        - vitb_dinov3_lora
        - vitb_imgnet_lora
        - convnext_dinov3_lora
        - convnext_imgnet_lora
    """

    if model_name == 'vitb_dinov2_lora':
        backbone = AutoModel.from_pretrained(
            "facebook/dinov2-base",
            attn_implementation="sdpa",
            dtype=torch.float32,
        ).float()
        head_in_features = 768
        model_family = 'vit'
        trainer_model_name = 'vitb_dinov2'
        use_wrapper = True

    elif model_name == 'vitb_dinov3_lora':
        backbone = AutoModel.from_pretrained(
            "facebook/dinov3-vitb16-pretrain-lvd1689m",
            attn_implementation="sdpa",
            dtype=torch.float32,
        ).float()
        head_in_features = 768
        model_family = 'vit'
        trainer_model_name = 'vitb_dinov3'
        use_wrapper = True

    elif model_name == 'convnext_dinov3_lora':
        backbone = AutoModel.from_pretrained(
            "facebook/dinov3-convnext-base-pretrain-lvd1689m",
            use_safetensors=True,
            dtype=torch.float32,
        ).float()
        head_in_features = 1024
        model_family = 'convnext'
        trainer_model_name = 'convnext_dinov3'
        use_wrapper = True

    elif model_name == 'convnext_imgnet_lora':
        backbone = AutoModel.from_pretrained(
            "facebook/convnext-base-224-22k",
            use_safetensors=True,
            dtype=torch.float32,
        ).float()
        head_in_features = 1024
        model_family = 'convnext'
        trainer_model_name = 'convnext_imgnet'
        use_wrapper = True

    elif model_name == 'vitb_imgnet_lora':
        backbone = timm.create_model(
            'vit_base_patch16_224_in21k',
            num_classes=num_classes,
            img_size=image_size,
            pretrained=True
        )
        model_family = 'vit'
        trainer_model_name = 'vitb_imgnet'
        use_wrapper = False

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    if use_wrapper:
        model = BackboneWithHead(
            backbone=backbone,
            head_in_features=head_in_features,
            num_classes=num_classes
        )

        target_modules = _infer_lora_target_modules(model.backbone, model_family)

        peft_config = LoraConfig(
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=target_modules,
        )

        model.backbone = get_peft_model(
            model.backbone,
            peft_config,
            autocast_adapter_dtype=False,
        )

        for p in model.head.parameters():
            p.requires_grad = True

        print("[LoRA] Trainable parameters in backbone:")
        model.backbone.print_trainable_parameters()

    else:
        target_modules = _infer_lora_target_modules(backbone, model_family)

        peft_config = LoraConfig(
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=target_modules,
            modules_to_save=["head"],
        )

        model = get_peft_model(
            backbone,
            peft_config,
            autocast_adapter_dtype=False,
        )

        for p in model.head.parameters():
            p.requires_grad = True

        print("[LoRA] Trainable parameters in full timm model:")
        model.print_trainable_parameters()

    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[LoRA] Total trainable parameters including head: {total_trainable:,}")

    return model, trainer_model_name