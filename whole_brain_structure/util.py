########## Import Libararies
import os
from collections import Counter
import pickle
from tqdm.notebook import tqdm, trange

import numpy as np
from skimage import measure
import scipy.ndimage

 
import open3d as o3d
import trimesh

import nibabel as nib
import csv
from compas.geometry import trimesh_remesh
from compas.datastructures import Mesh

########## Create boundary_pt and its texture into pt_{}.pickle file
def marching_cubes_3d(data):
  vertices, triangles, _, _ = measure.marching_cubes(data, level=0)
  return vertices, triangles

def voxel_to_mesh(voxel):
  # Convert voxel to mesh
  vertices, triangles = marching_cubes_3d(voxel)
  tri = triangles.copy()
  triangles[:, 0] = tri[:, 2]
  triangles[:, 2] = tri[:, 0]
  mesh = o3d.geometry.TriangleMesh()
  mesh.vertices = o3d.utility.Vector3dVector(vertices)
  mesh.triangles = o3d.utility.Vector3iVector(triangles)
  return mesh

def voxel_to_bd_pcd(voxel, threshold = 8):
  kernel = np.ones((2,2,2))
  voxel_bd = scipy.ndimage.convolve((voxel).astype(int), kernel, mode='constant', cval=0.0)
  voxel_v = np.argwhere((voxel_bd<threshold)&(voxel_bd>0)).astype(float) + 0.5
  voxel_pcd = o3d.geometry.PointCloud()
  voxel_pcd.points = o3d.utility.Vector3dVector(voxel_v)
  return voxel_pcd

def pcd_to_orgin(pcd):
  pcd_p = np.asarray(pcd.points)

  for i in range(3):
    min_v = np.mean(pcd_p[:,i])
    pcd_p[:,i] -= min_v
  pcd.points = o3d.utility.Vector3dVector(pcd_p)
  return pcd

def mesh_to_np(mesh):
  ############## mesh to numpy occupancy ##############
  mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

  # Create a scene and add the triangle mesh
  scene = o3d.t.geometry.RaycastingScene()
  _ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh

  occupancy = scene.compute_occupancy(queries)
  return occupancy.numpy()

def generate_uniform_samples(mesh, num_samples):
  # Convert Open3D Triangle Mesh to NumPy arrays
  
  vertices = np.asarray(mesh.vertices)
  triangles = np.asarray(mesh.triangles)
  mesh.compute_triangle_normals()
  tri_normals = np.asarray(mesh.triangle_normals)
  
  # Compute face areas
  face_areas = np.linalg.norm(np.cross(vertices[triangles[:, 0]] - vertices[triangles[:, 1]],
                                        vertices[triangles[:, 0]] - vertices[triangles[:, 2]]), axis=1) / 2.0

  # Normalize face areas to get probabilities for sampling
  probabilities = face_areas / np.sum(face_areas)

  # Sample faces based on the probabilities
  sampled_faces = np.random.choice(len(triangles), size=num_samples, p=probabilities, replace=True)

  # Sample points uniformly on the selected faces
  sampled_points = []
  for face_idx in sampled_faces:
    face = triangles[face_idx]
    barycentric_coords = np.random.rand(2, 1)
    barycentric_coords /= np.sum(barycentric_coords, axis=0)
    sampled_point = (1 - barycentric_coords[0] - barycentric_coords[1]) * vertices[face[0]] + \
                      barycentric_coords[0] * vertices[face[1]] + barycentric_coords[1] * vertices[face[2]]
    sampled_points.append(sampled_point)

  sampled_points = np.vstack(sampled_points)

  # Compute normals at the sampled points
  sampled_normals= tri_normals[sampled_faces]
  # print(sampled_normals.shape, sampled_points.shape)
  
  return sampled_points, sampled_normals

def voxel_2_gridpoint(vox): 
	## Generate Boundary_pt
	# print(np.sum(vox==0),np.sum(vox!=0))
	k1 = np.array([1,1]) #the kernel along the 1st dimension
	k2 = k1 #the kernel along the 2nd dimension
	k3 = k1
	# Convolve over all three axes in a for loop
	out = vox.copy()
	for i, k in enumerate((k1, k2, k3)):
		out = scipy.ndimage.convolve1d(out, k, axis=i)
	return out

########## Create boundary_pt and its texture into pt_{}.pickle file
def LV_smoothing(vox): 
	## Generate Boundary_pt
	# print(np.sum(vox==0),np.sum(vox!=0))
	k1 = np.array([1, 1, 1]) #the kernel along the 1st dimension
	k2 = k1 #the kernel along the 2nd dimension
	k3 = k1
	# Convolve over all three axes in a for loop
	out = vox.copy()
	for i, k in enumerate((k1, k2, k3)):
		out = scipy.ndimage.convolve1d(out, k, axis=i)
	return out

def edge_adjusted_mesh(mesh, target_length = 1):

  mesh_tri = Mesh.from_vertices_and_faces(vertices=np.asarray(mesh.vertices).tolist(),
                        faces=np.asarray(mesh.triangles).tolist())                       
  vert, fac = trimesh_remesh(mesh_tri.to_vertices_and_faces(), target_edge_length=target_length)
  print(len(vert))
  mesh_avg= o3d.geometry.TriangleMesh()
  mesh_avg.vertices = o3d.utility.Vector3dVector(np.asarray(vert))
  mesh_avg.triangles = o3d.utility.Vector3iVector(np.asarray(fac))
  return mesh_avg

def sdf_query(mesh, query_points):

  mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

  # Create a scene and add the triangle mesh
  scene = o3d.t.geometry.RaycastingScene()
  _ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh
  
  sdf = scene.compute_signed_distance(query_points)
  # print(query_points.shape, occupancy.shape)
  # We can visualize a slice of the distance field directly with matplotlib
  # plt.imshow(occupancy.numpy()[:, :, 70])
  #print(np.shape(sdf.numpy()))
  return sdf.numpy()

def find_nearest_points_indices(point_cloud1, point_cloud2):
    # Calculate the distance matrix
    dist_matrix = np.linalg.norm(point_cloud1[:, None, :] - point_cloud2[None, :, :], axis=-1)
    
    # Find the index of the nearest point for each point in point_cloud1
    nearest_indices = np.argmin(dist_matrix, axis=1)
    
    return nearest_indices

def generate_template_mesh_pcd(mesh, pt_num=3000, edge_length=2.8):
   
	temp_model_com = o3d.geometry.TriangleMesh.filter_smooth_taubin(mesh, number_of_iterations=5)
	mesh_tri = Mesh.from_vertices_and_faces(vertices=np.asarray(temp_model_com.vertices).tolist(),
												faces=np.asarray(temp_model_com.triangles).tolist())
											
	vert, fac = trimesh_remesh(mesh_tri.to_vertices_and_faces(), target_edge_length=edge_length)
	print(len(vert))

	temp_mesh= o3d.geometry.TriangleMesh()
	temp_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vert))
	temp_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(fac))

	points, normals = generate_uniform_samples(temp_mesh, pt_num)

	temp_pcd = o3d.geometry.PointCloud()
	temp_pcd.points = o3d.utility.Vector3dVector(points)
	temp_pcd.normals = o3d.utility.Vector3dVector(normals)

	points_0, normals_0 = generate_uniform_samples(mesh, pt_num)

	return temp_mesh, temp_pcd


def read_sub_from_csv(file_path, flag = "LBC"):
    data = []
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        for index, row in enumerate(csv_reader):
            if index==0: continue
            else:
                if flag in row[0]:
                    id= row[0]
                    data.append(id)
    return data

def read_dict_from_csv(file_path, flag = "LBC"):
    data = {}
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        for index, row in enumerate(csv_reader):
            if index==0: continue
            else:
                if flag in row[0]:
                    id= row[0]
                    data[id] = row
    return data

def pt_to_tex(boundary_pt, aseg_path, tex_label = [10, 11, 43]):
  
    aseg_peri = np.array(nib.load(aseg_path).dataobj)
    aseg_peri[(aseg_peri!=tex_label[0])&(aseg_peri!=tex_label[1])&(aseg_peri!=tex_label[2])]=0
    aseg_peri[aseg_peri==tex_label[0]]=1
    aseg_peri[aseg_peri==tex_label[1]]=2
    aseg_peri[aseg_peri==tex_label[2]]=3
    texture = np.zeros(boundary_pt.shape)
    print(texture.shape)
    
    for n in range(boundary_pt.shape[0]):
      a= []
      x, y, z = (boundary_pt[n][0]), (boundary_pt[n][1]), (boundary_pt[n][2])
      peri_list = [2.5, 1.5, 0.5, -0.5, -1.5, -2.5]
      for i in peri_list:
        for j in peri_list:
          for k in peri_list:
            tex = aseg_peri[int(x+i),int(y+j),int(z+k)]
            if tex!=0:
              a.append(tex)
      if a:
        counts = Counter(a)
        most_common_value = max(counts, key=lambda x: counts[x] if (x!=0) else -1)
        texture[n]=int(most_common_value)

    return  texture

def index_of_tex(texture_new, tri, tex_label):
    tex_index = np.argwhere(texture_new[:,0]==tex_label).flatten()
    mask = np.isin(tri, tex_index).any(axis=1)
    indices = np.where(mask)[0]
    return indices
  
def regist_mesh_pcd(src_pcd, tar_pcd, src_mesh=None, return_reg_p2p=False, normal=True):
  if normal ==False:
      reg_p2p = o3d.pipelines.registration.registration_icp(  # source, target
		src_pcd, tar_pcd, max_correspondence_distance=10,  # Adjust the distance threshold as needed
		estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
		criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))
  else:
    reg_p2p = o3d.pipelines.registration.registration_icp(  # source, target
      src_pcd, tar_pcd, max_correspondence_distance=10,  # Adjust the distance threshold as needed
      estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
      criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))
    
  src_pcd.transform(reg_p2p.transformation)
  if src_mesh: src_mesh.transform(reg_p2p.transformation)
  if return_reg_p2p:
    return reg_p2p.transformation
  