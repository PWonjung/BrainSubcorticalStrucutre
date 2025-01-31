import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class PointNetFeat(nn.Module):
    """
    Corresponds to the part that extracts max-pooled features.
    """
    def __init__(
        self,
        input_transform: bool = False,
        feature_transform: bool = False,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        # point-wise mlp
        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64))
        self.conv3 = nn.Sequential(nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64))
        self.conv4 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv5 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))


    def forward(self, pointcloud):
        """
        Input:
            pointcloud: [B,N,3]
        Output:
            logits: [B, num_classes]
        """
        x = pointcloud.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        concat_feat = x 
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))                                                
        
        x = torch.max(x, 2)[0]
        
        return x, concat_feat


class PointNetCls(nn.Module):
    def __init__(self, num_classes, input_transform, feature_transform):
        super().__init__()
        self.num_classes = num_classes
        
        # extracts max-pooled features
        self.pointnet_feat = PointNetFeat(input_transform, feature_transform)
        
        # returns the final logits from the max-pooled features.
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256,  self.num_classes),
        )

    def forward(self, pointcloud):
        x, trans, feat_trans,_ = self.pointnet_feat(pointcloud)

        x = self.fc(x)  
        return x, trans, feat_trans


class PointNetAutoEncoder(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.pointnet_feat = PointNetFeat()

        # Decoder is just a simple MLP that outputs N x 3 (x,y,z) coordinates.
        self.decoder = nn.Sequential(
            nn.Linear(1024, num_points // 4),
            nn.BatchNorm1d(num_points // 4),
            nn.ReLU(),
            nn.Linear(num_points // 4, num_points // 2),
            nn.BatchNorm1d(num_points // 2),
            nn.ReLU(),
            nn.Linear(num_points // 2, num_points),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(num_points),
            nn.ReLU(),
            nn.Linear(num_points, num_points * 3),
        )

    def forward(self, pointcloud):
        B, N, _ = pointcloud.shape

        x, _, _,_ = self.pointnet_feat(pointcloud)
        x = self.decoder(x)
        x = x.reshape(B, N, 3)

        return x


class PointNetOpt(nn.Module):
    def __init__(self, num_classes, input_transform, feature_transform):
        # TODO: Implement this
        super().__init__()
        self.num_classes = num_classes
        self.feature_transform= feature_transform
        # extracts max-pooled features
        self.pointnet_feat = PointNetFeat(input_transform, feature_transform)

        # point-wise mlp
        self.conv1 = nn.Sequential(nn.Conv1d(1088, 512, 1), nn.BatchNorm1d(512))
        self.conv2 = nn.Sequential(nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256))
        self.conv3 = nn.Sequential(nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128))
        
        self.conv4 = nn.Sequential(nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128))
        #self.conv4 = nn.Sequential(nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.Dropout(p=0.3)) #Dropout

        self.conv5 = nn.Sequential(nn.Conv1d(128, self.num_classes, 1), nn.BatchNorm1d(self.num_classes))

    def forward(self, pointcloud):

        """
        Input:
            pointcloud: [B,N,3]
        Output:
            logits: [B,50,N] | 50: # of point labels
        """
        #print(pointcloud.size())
        
        x, concat_feat = self.pointnet_feat(pointcloud)
        
        # repeat global features N times
        x = x.repeat(pointcloud.size()[1],1,1) 
        x= x.transpose(0,1)
        x= x.transpose(1,2)

        # concatenate local and global features
        x = torch.cat([concat_feat, x], dim=1)
        # MLP
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = (self.conv5(x))

        #print(x.size()) # to check output size
        
        return x
        
        pass

