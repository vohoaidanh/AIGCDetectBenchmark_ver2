import functools
import torch
import torch.nn as nn
from networks.resnet import resnet50

from networks.base_model import BaseModel, init_weights

from util import get_model

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            # 会要用预训练的模型
            self.model = get_model(opt)


        if not self.isTrain or opt.continue_train:
            self.model = resnet50(num_classes=1)

        if self.isTrain:
            self.loss_fn = self.set_loss(opt)
            # initialize optimizers
            
            if self.opt.detect_method in ['CNNSpot_CAM', 'Resnet_Metric', 'Resnet_Multiscale']:
                params = self.model.get_parameters_to_optimize()
            
            elif self.opt.detect_method == "UnivFD" and self.opt.fix_backbone:
                params = []
                for name, p in self.model.named_parameters():
                    if  name=="fc.weight" or name=="fc.bias": 
                        params.append(p) 
                    else:
                        p.requires_grad = False
            
            else:
                params = self.model.parameters()
            
            params = [p for p in params if p.requires_grad]
            
            if opt.optim == 'adam':
                if self.opt.detect_method == "UnivFD":
                    self.optimizer = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
                else:
                    self.optimizer = torch.optim.Adam(params,
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(params,
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        self.model.to(self.device)

    
    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, input):
     
        if self.opt.detect_method == "Fusing":
            self.input_img = input[0] # (batch_size, 6, 3, 224, 224)
            self.cropped_img = input[1].to(self.device)
            self.label = input[2].to(self.device).float() #(batch_size)
            self.scale = input[3].to(self.device).float()
            
        elif self.opt.detect_method == "Combine":
            self.input = input[0].to(self.device)
            self.input2 = input[1].to(self.device)
            self.label = input[2].to(self.device).float() #(batch_size)
        
        elif self.opt.detect_method == "CNNSpot_CAM":
            self.input = input[0].to(self.device)
            self.input2 = input[1].to(self.device)
            self.label = input[2].to(self.device).float() #(batch_size)
            
        elif self.opt.detect_method == "Resnet_Metric":
            self.input = (input[0][0].to(self.device), input[0][1].to(self.device))
            self.label = input[1].to(self.device).float() #(batch_size)
            
        else:
            self.input = input[0].to(self.device)
            self.label = input[1].to(self.device).float()


    def forward(self):
        if self.opt.detect_method == "Fusing":
            self.output = self.model(self.input_img, self.cropped_img, self.scale)
        elif self.opt.detect_method == "UnivFD":
            self.output = self.model(self.input)
            self.output = self.output.view(-1).unsqueeze(1)
        elif self.opt.detect_method == "Combine":
            self.output = self.model(self.input, self.input2)
        elif self.opt.detect_method == "CNNSpot_CAM" :
            self.output = self.model(self.input, self.input2)
        else: 
            self.output = self.model(self.input)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)
    
    def set_loss(self, opt):
        if opt.detect_method == "Resnet_Metric":
            from networks.resnet_metric import cosine_similarity_loss
            loss_fn = cosine_similarity_loss
            return loss_fn
        return nn.BCEWithLogitsLoss()

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

