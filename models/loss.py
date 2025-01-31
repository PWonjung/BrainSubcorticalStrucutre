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
from models.loss_3d import mesh_edge_var_loss
from models.point_mesh_loss2 import point_mesh_face_distance2#point_mesh_face_weighted_distance
# from models.normal_L2_loss import normal_L2_loss


def criterion(verts, target, faces):
    loss=0.0

    # cf loss
    pred_mesh = Meshes(verts=list(verts), faces=list(faces))
    pred_points = sample_points_from_meshes(pred_mesh, target.shape[1])
    chamfer_loss =  chamfer_distance(pred_points, target)[0]
    
    # point-mesh loss
    pointclouds = Pointclouds(points=target)
    point_mesh_dist_loss = point_mesh_face_distance(pred_mesh, pointclouds)

    # Regularization loss
    edge_loss = mesh_edge_var_loss(pred_mesh)
    laplacian_loss =  mesh_laplacian_smoothing(pred_mesh, method="uniform")
    normal_consistency_loss = mesh_normal_consistency(pred_mesh)
    edge_loss = mesh_edge_var_loss(pred_mesh)
    
    loss =  3*  (point_mesh_dist_loss)\
    + 0.5 * (chamfer_loss)\
    + 1500 * (edge_loss)\
    + 5 * (laplacian_loss)\
    + 1 * (normal_consistency_loss)\

    log = {"loss": loss,
        "point_mesh_dist_loss": point_mesh_dist_loss.detach(),
        "chamfer_loss": chamfer_loss.detach(),
        "edge_loss": edge_loss.detach(),
        "laplacian_loss": laplacian_loss.detach(),
        "normal_consistency_loss": normal_consistency_loss.detach()
        }
    
    return loss, log