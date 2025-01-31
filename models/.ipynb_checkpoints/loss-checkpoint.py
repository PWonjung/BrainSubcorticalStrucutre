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
from models.point_mesh_loss import point_mesh_face_weighted_distance
# from models.normal_L2_loss import normal_L2_loss

def criterion(verts, target, faces, tex_faces_train= None, tex_target_pt_train= None, loss_type=None):

    cf_loss =0.0
    loss=0.0
    cf_log = {}
    pred_mesh = Meshes(verts=list(verts), faces=list(faces))
    pred_points = sample_points_from_meshes(pred_mesh, target.shape[1])
    chamfer_loss =  chamfer_distance(pred_points, target)[0]
    laplacian_loss =  mesh_laplacian_smoothing(pred_mesh, method="uniform")
    normal_consistency_loss = mesh_normal_consistency(pred_mesh)
    edge_loss = mesh_edge_loss(pred_mesh, 1.5)
    
    pointclouds = Pointclouds(points=target)

    point_mesh_dist_loss = point_mesh_face_distance(pred_mesh, pointclouds)
    
    loss = 0.5 * laplacian_loss + 0.5 * edge_loss + normal_consistency_loss + point_mesh_dist_loss + 0.5 * chamfer_loss
    
    tex_loss = 0
    for i in range(len(tex_target_pt_train)):
        pred_mesh = Meshes(verts=list(verts), faces=list(tex_faces_train[i]))
        target_sample  = tex_target_pt_train[i]
        pointclouds = Pointclouds(points=target_sample)
        point_mesh_dist_loss = point_mesh_face_weighted_distance(pred_mesh, pointclouds)
        tex_loss += point_mesh_dist_loss / len(tex_target_pt_train)
        cf_log[str(i)]=tex_loss
        
    loss += tex_loss
    log = {"loss": loss,
           "chamfer_loss": chamfer_loss.detach(),
            "tex_loss": tex_loss,
           "normal_consistency_loss": normal_consistency_loss.detach(),
           "edge_loss": edge_loss.detach(),
           "laplacian_loss": laplacian_loss.detach(),
           "point_mesh_dist_loss": point_mesh_dist_loss.detach()
          }
    
    return loss, log, cf_log


def tex_loss(bf_vert, ve6rt_tex, target_tex)-> torch.Tensor:
    
    bf_tex = torch.zeros(bf_vert.shape).cuda()
    print(f"{bf_vert.requires_grad=}, {bf_tex.requires_grad=}")
    
    for n in range(bf_vert.shape[0]):
        a= []
        pt_x = int(bf_vert[n,0])
        pt_y = int(bf_vert[n,1])
        pt_z = int(bf_vert[n,2])

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    tex = target_tex[i+pt_x,j+pt_y,k+pt_z]
                    a.append(tex)
        counts = Counter(a)
        most_common_value = max(counts, key=lambda x: counts[x] if (x != 4 and x!=5 and x!=31 and x!=0) else -1)
        #if most_common_value == 4: print("4 is here!")
        bf_tex[n] = int(most_common_value)
    
    accuracy = torch.sum(vert_tex[:, 0] == bf_tex[:,0])/bf_tex.shape[0]


    return accuracy
    return loss, log
"""
def lvhippo_loss(verts, lv_faces, hippo_faces, lv_target, hippo_target):
    lv_pred_mesh = Meshes(verts=list(verts), faces=list(lv_faces))
    hippo_pred_mesh = Meshes(verts=list(verts), faces=list(hippo_faces))
    
    lv_pcd = Pointclouds(points=lv_target)
    hippo_pcd = Pointclouds(points=hippo_target)

    lv_point_mesh_dist_loss = point_mesh_face_distance(lv_pred_mesh, lv_pcd)
    hippo_point_mesh_dist_loss = point_mesh_face_distance(hippo_pred_mesh, hippo_pcd)
    
    #loss = (lv_point_mesh_dist_loss+hippo_point_mesh_dist_loss)
    
    loss =lv_point_mesh_dist_loss        
    
    log = {"loss": loss.detach(),
          }

    return loss, log

"""
def lvhippo_loss(verts, lv_faces, hippo_faces, lv_target, hippo_target,epoch=0):
    lv_pred_mesh = Meshes(verts=list(verts), faces=list(lv_faces))
    hippo_pred_mesh = Meshes(verts=list(verts), faces=list(hippo_faces))
    
    chamfer_losses=[]
    
    lv_laplacian_loss = mesh_laplacian_smoothing(lv_pred_mesh, method="uniform")
    hippo_laplacian_loss = mesh_laplacian_smoothing(hippo_pred_mesh, method="uniform")

    lv_normal_consistency_loss = mesh_normal_consistency(lv_pred_mesh)
    hippo_normal_consistency_loss = mesh_normal_consistency(hippo_pred_mesh)

    
    #lv_edge_loss = mesh_edge_loss(lv_pred_mesh,2.8)
    #hippo_edge_loss = mesh_edge_loss(hippo_pred_mesh,2.8)
    lv_edge_loss = mesh_edge_var_loss(lv_pred_mesh)
    hippo_edge_loss = mesh_edge_var_loss(hippo_pred_mesh)

    lv_pcd = Pointclouds(points=lv_target)
    hippo_pcd = Pointclouds(points=hippo_target)

    lv_point_mesh_dist_loss = point_mesh_face_weighted_distance(lv_pred_mesh, lv_pcd)
    hippo_point_mesh_dist_loss = point_mesh_face_weighted_distance(hippo_pred_mesh, hippo_pcd)
    
    #normal_loss = normal_L2_loss(hippo_pred_mesh)
    loss =  2* (lv_laplacian_loss+hippo_laplacian_loss-180)\
    + 1500* (lv_edge_loss+hippo_edge_loss)\
    + (lv_normal_consistency_loss + hippo_normal_consistency_loss)\
    + (lv_point_mesh_dist_loss+hippo_point_mesh_dist_loss)*3
    cf_loss =0.0
    cf_log = {}

    
            
    lv_chamfer_num=2000
    lv_pred_points = sample_points_from_meshes(lv_pred_mesh, lv_chamfer_num)
    lv_random_indices = torch.randperm(lv_target.size(1))[:lv_chamfer_num]
    lv_target_points = lv_target[:, lv_random_indices, :]
    lv_chamfer_loss, _ = chamfer_distance(lv_pred_points, lv_target_points)
    
    hippo_chamfer_num=200
    hippo_pred_points = sample_points_from_meshes(hippo_pred_mesh, hippo_chamfer_num)
    hippo_random_indices = torch.randperm(hippo_target.size(1))[:hippo_chamfer_num]
    hippo_target_points = hippo_target[:,hippo_random_indices, :]
    hippo_chamfer_loss, _ = chamfer_distance(hippo_pred_points, hippo_target_points)
    
    loss += (lv_chamfer_loss+hippo_chamfer_loss)*0.5
        
    if epoch<5000: loss = lv_point_mesh_dist_loss+hippo_point_mesh_dist_loss
        
    log = {"loss": loss.detach(),
           "chamfer_loss": lv_chamfer_loss.detach()+hippo_chamfer_loss.detach(), 
           "normal_consistency_loss": lv_normal_consistency_loss.detach()+hippo_normal_consistency_loss.detach(),
           "edge_loss": lv_edge_loss.detach()+hippo_edge_loss.detach(),
           "laplacian_loss": lv_laplacian_loss.detach()+hippo_laplacian_loss.detach()-180,
           "point_mesh_dist_loss": lv_point_mesh_dist_loss.detach()+hippo_point_mesh_dist_loss.detach()
          }
#     loss += lv_point_mesh_dist_loss+hippo_point_mesh_dist_loss

    print(loss)
    return loss, log

def onlylv_loss(verts, lv_faces, hippo_faces, lv_target, hippo_target):
    lv_pred_mesh = Meshes(verts=list(verts), faces=list(lv_faces))
    hippo_pred_mesh = Meshes(verts=list(verts), faces=list(hippo_faces))
    
    chamfer_losses=[]
    
    lv_laplacian_loss = mesh_laplacian_smoothing(lv_pred_mesh, method="uniform")
    hippo_laplacian_loss = 0

    lv_normal_consistency_loss = mesh_normal_consistency(lv_pred_mesh)
    hippo_normal_consistency_loss = 0

    
    #lv_edge_loss = mesh_edge_loss(lv_pred_mesh,2.8)
    #hippo_edge_loss = mesh_edge_loss(hippo_pred_mesh,2.8)
    lv_edge_loss = mesh_edge_var_loss(lv_pred_mesh)
    hippo_edge_loss =0

    lv_pcd = Pointclouds(points=lv_target)
    hippo_pcd = Pointclouds(points=hippo_target)

    lv_point_mesh_dist_loss = point_mesh_face_weighted_distance(lv_pred_mesh, lv_pcd)
    hippo_point_mesh_dist_loss = 0
    
    #normal_loss = normal_L2_loss(hippo_pred_mesh)
    loss =  5* (lv_laplacian_loss)\
    + 1500* (lv_edge_loss)\
    + (lv_normal_consistency_loss)\
    + (lv_point_mesh_dist_loss)*3
    cf_loss =0.0
    cf_log = {}

    
            
    lv_chamfer_num=2000
    lv_pred_points = sample_points_from_meshes(lv_pred_mesh, lv_chamfer_num)
    lv_random_indices = torch.randperm(lv_target.size(1))[:lv_chamfer_num]
    lv_target_points = lv_target[:, lv_random_indices, :]
    lv_chamfer_loss, _ = chamfer_distance(lv_pred_points, lv_target_points)
    
    hippo_chamfer_num=200
    hippo_pred_points = sample_points_from_meshes(hippo_pred_mesh, hippo_chamfer_num)
    hippo_random_indices = torch.randperm(hippo_target.size(1))[:hippo_chamfer_num]
    hippo_target_points = hippo_target[:,hippo_random_indices, :]
    hippo_chamfer_loss, _ = chamfer_distance(hippo_pred_points, hippo_target_points)
    hippo_chamfer_loss =0
    
    loss += (lv_chamfer_loss)*0.5
        
    
    log = {"loss": loss.detach(),
           "chamfer_loss": lv_chamfer_loss.detach(), 
           "normal_consistency_loss": lv_normal_consistency_loss.detach(),
           "edge_loss": lv_edge_loss.detach(),
           "laplacian_loss": lv_laplacian_loss.detach(),
           "point_mesh_dist_loss": lv_point_mesh_dist_loss.detach()
          }
#     loss = lv_point_mesh_dist_loss

    print(loss)
    return loss, log

def onlyhippo_loss(verts, lv_faces, hippo_faces, lv_target, hippo_target):
    lv_pred_mesh = Meshes(verts=list(verts), faces=list(lv_faces))
    hippo_pred_mesh = Meshes(verts=list(verts), faces=list(hippo_faces))
    
    chamfer_losses=[]
    
    lv_laplacian_loss = 0
    hippo_laplacian_loss = mesh_laplacian_smoothing(hippo_pred_mesh, method="uniform")

    lv_normal_consistency_loss = 0
    hippo_normal_consistency_loss = mesh_normal_consistency(hippo_pred_mesh)

    
    #lv_edge_loss = mesh_edge_loss(lv_pred_mesh,2.8)
    #hippo_edge_loss = mesh_edge_loss(hippo_pred_mesh,2.8)
    lv_edge_loss = 0
    hippo_edge_loss =mesh_edge_var_loss(hippo_pred_mesh)

    lv_pcd = Pointclouds(points=lv_target)
    hippo_pcd = Pointclouds(points=hippo_target)

    lv_point_mesh_dist_loss = 0
    hippo_point_mesh_dist_loss = point_mesh_face_weighted_distance(hippo_pred_mesh, hippo_pcd)
    
    #normal_loss = normal_L2_loss(hippo_pred_mesh)
    loss =  5* (hippo_laplacian_loss)\
    + 1500* (hippo_edge_loss)\
    + (hippo_normal_consistency_loss)\
    + (hippo_point_mesh_dist_loss)*3
    cf_loss =0.0
    cf_log = {}

    
            
    lv_chamfer_num=2000
    lv_pred_points = sample_points_from_meshes(lv_pred_mesh, lv_chamfer_num)
    lv_random_indices = torch.randperm(lv_target.size(1))[:lv_chamfer_num]
    lv_target_points = lv_target[:, lv_random_indices, :]
    lv_chamfer_loss, _ = chamfer_distance(lv_pred_points, lv_target_points)
    lv_chamfer_loss =0
    hippo_chamfer_num=200
    hippo_pred_points = sample_points_from_meshes(hippo_pred_mesh, hippo_chamfer_num)
    hippo_random_indices = torch.randperm(hippo_target.size(1))[:hippo_chamfer_num]
    hippo_target_points = hippo_target[:,hippo_random_indices, :]
    hippo_chamfer_loss, _ = chamfer_distance(hippo_pred_points, hippo_target_points)
    
    
    loss += (hippo_chamfer_loss)*0.5
        
    
    log = {"loss": loss.detach(),
           "chamfer_loss": hippo_chamfer_loss.detach(), 
           "normal_consistency_loss": hippo_normal_consistency_loss.detach(),
           "edge_loss": hippo_edge_loss.detach(),
           "laplacian_loss": hippo_laplacian_loss.detach(),
           "point_mesh_dist_loss": hippo_point_mesh_dist_loss.detach()
          }
#     loss = lv_point_mesh_dist_loss

    print(loss)
    return loss, log