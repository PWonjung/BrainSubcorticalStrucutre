import torch.nn as nn
import torch 
import torch.nn.functional as F 

from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (chamfer_distance,  mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency, point_mesh_face_distance)

import numpy as np
from itertools import product, combinations, chain
from scipy.spatial import ConvexHull

import time 
from collections import Counter
import torch.nn.functional as F

def criterion(verts, target, faces, tex=None, target_tex=None, tri_tex = None, tex_index=None, loss_type=None):
         

#     chamfer_loss = torch.tensor(0).float().cuda()
#     edge_loss = torch.tensor(0).float().cuda()
#     laplacian_loss = torch.tensor(0).float().cuda()
#     normal_consistency_loss = torch.tensor(0).float().cuda()  



    pred_mesh = Meshes(verts=list(verts), faces=list(faces))
    pred_points = sample_points_from_meshes(pred_mesh, 3000)

    chamfer_loss =  chamfer_distance(pred_points, target)[0]
    laplacian_loss =   mesh_laplacian_smoothing(pred_mesh, method="uniform")
    normal_consistency_loss = mesh_normal_consistency(pred_mesh) 
    edge_loss = mesh_edge_loss(pred_mesh)
    
    pointclouds = Pointclouds(points=target)

    point_mesh_dist_loss = point_mesh_face_distance(pred_mesh, pointclouds)
    chamfer_loss=0
    loss = 0.3 * laplacian_loss + 0.1* edge_loss + 0.3 * normal_consistency_loss+ point_mesh_dist_loss+chamfer_loss
       
    log = {"loss": loss.detach(),
           #"chamfer_loss": chamfer_loss.detach(), 
           "normal_consistency_loss": normal_consistency_loss.detach(),
           "edge_loss": edge_loss.detach(),
           "laplacian_loss": laplacian_loss.detach(),
           "point_mesh_dist_loss": point_mesh_dist_loss.detach()
          }
    
    return loss, log