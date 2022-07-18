import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
# from torchvision.models import 
import torchvision.models as models
import torchmetrics 
from .models import FaceLandmarkNet

class FaceLandmarkTask(pl.LightningModule):
    def __init__(self, pretrained=True, in_chan=1, num_pts=136, lr=0.001, **kwargs):
        super().__init__()
        
        self.model = FaceLandmarkNet(in_chan=in_chan, num_pts=num_pts)
        
        self.trn_acc: torchmetrics.Accuracy = torchmetrics.Accuracy()
        self.trn_loss: torchmetrics.AverageMeter  = torchmetrics.MeanMetric()
        self.val_acc: torchmetrics.Accuracy = torchmetrics.Accuracy()
        self.val_loss: torchmetrics.AverageMeter  = torchmetrics.MeanMetric()
        
        self.criterion = nn.MSELoss()
        self.save_hyperparameters()
        self.save_hyperparameters(kwargs)
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
    
    def forward(self, x):
        return self.model(x)
    
    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()
        
    def shared_step(self, batch, batch_idx):
        images, key_pts = batch
        
        # flatten pts
        key_pts = key_pts.view(key_pts.size(0), -1)
        
        # convert variables to floats for regression loss
        key_pts = key_pts.type(torch.FloatTensor)
        images = images.type(torch.FloatTensor)
        
        preds = self.model(images) # align with Attention.forward
        loss = self.criterion(preds, key_pts)
        return loss, preds, key_pts
    
    def training_step(self, batch, batch_idx):
        loss, preds, labels = self.shared_step(batch, batch_idx)
        trn_acc = self.trn_acc(preds, labels)
        trn_loss = self.trn_loss(loss)
        
        self.log('trn_step_loss', trn_loss, prog_bar=True, logger=True)
        self.log('trn_step_acc', trn_acc,  prog_bar=True, logger=True)
        
        return loss
        
    def training_epoch_end(self, outs):
        self.log('trn_epoch_acc', self.trn_acc.compute(), logger=True)
        self.log('trn_epoch_loss', self.trn_loss.compute(), logger=True)
        
    
    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.shared_step(batch, batch_idx)
        val_acc = self.val_acc(preds, labels)
        val_loss = self.val_loss(loss)
        
        self.log('val_step_loss', val_loss, prog_bar=True, logger=True)
        self.log('val_step_acc', val_acc,  prog_bar=True, logger=True)
        
        return loss
    
    def validation_epoch_end(self, outs):
        self.log('val_epoch_acc', self.val_acc.compute(), logger=True)
        self.log('val_epoch_loss', self.val_loss.compute(), logger=True)
    
    
    
if __name__ == '__main__':
    import sys
    import os
    import torch

    current_dir = os.path.dirname(__file__)
    sys.path.insert(0, current_dir)
    mvn2 = FaceLandmarkTask()
    batch = torch.rand(2,3,224,224)
    result = mvn2(batch)
    print(result)