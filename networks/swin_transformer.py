# -*- coding: utf-8 -*-

from torchvision import models
import torch.nn as nn

__all__ = ['swin_transformer_model']


def swin_transformer_model(pretrained=True, num_classes=10):
    model = models.swin_t(weights='DEFAULT')
    #print(model)
    if pretrained:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    if not pretrained:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
    
    model.head = nn.Linear(
        in_features=768, 
        out_features=num_classes, 
        bias=True
    )
    return model


if __name__ == '__main__':
    import torch
    model = swin_transformer_model(pretrained=True, num_classes=1)
    input_tensor = torch.rand((2,3,224,224))
    model(input_tensor)

    sum(p.numel() for p in model.parameters())







