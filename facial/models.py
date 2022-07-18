## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torchvision.models as models

class FaceLandmarkNet(nn.Module):
    def __init__(self, in_chan=1, num_pts=136):
        super(FaceLandmarkNet, self).__init__()
        # Covolutional Layers
        self.conv1 = nn.Conv2d(in_chan, 32, 4) 
        self.conv2 = nn.Conv2d(32, 64, 3) 
        self.conv3 = nn.Conv2d(64, 128, 2) 
        self.conv4 = nn.Conv2d(128, 256, 1)

        # Maxpooling Layer	(for all)	
        self.pool = nn.MaxPool2d(2, 2)
		
		# Dropout (for all)	
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.2)
        self.drop3 = nn.Dropout(0.3)
        self.drop4 = nn.Dropout(0.4)
        self.drop5 = nn.Dropout(0.5)
        self.drop6 = nn.Dropout(0.6)
		
		# Fully Connected Layers (fc)
        self.fc1 = nn.Linear(in_features=43264, out_features=1000)  #torch.Size([10, 36864]) => in_features:  = 36864
        self.fc2 = nn.Linear(in_features=1000, out_features=1000)  
        self.fc3 = nn.Linear(in_features=1000, out_features=num_pts)    #68 keypoints with x and y coordinate => out_features: 136
        
    def forward(self, x):
        x = self.drop1(self.pool(F.relu(self.conv1(x))))
        x = self.drop2(self.pool(F.relu(self.conv2(x))))
        x = self.drop3(self.pool(F.relu(self.conv3(x))))
        x = self.drop4(self.pool(F.relu(self.conv4(x))))

		# Flattening the layer
        x = x.view(x.size(0), -1)
		
        # print("in_features size: ", x.size(1))
        x = self.drop5(F.relu(self.fc1(x)))
        x = self.drop6(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x