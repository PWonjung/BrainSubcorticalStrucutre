import cupy as cp 
from tqdm import tqdm
from numba import cuda

  
@cuda.jit
def pt_to_texture(pt, seg_label, pt_tex, CE):
  # input pt: n*3 points
  # input seg_label: w*h*d
  # output pt_tex: n size tex label
  P=cuda.grid(1)

  if P > pt.shape[0]:
    return
  pt_x = int(pt[P,0])
  pt_y = int(pt[P,1])
  pt_z = int(pt[P,2])
  for i in range(2):
    for j in range(2):
      #for k in [pt_z,pt_z+1]:
      if pt_tex[P]==seg_label[i+pt_x,j+pt_y]: CE[P]=1 
  #cand = seg_label[pt_x:pt_x+2, pt_y:pt_y+2]#, pt_z:pt_z+2]
  #if pt_tex[P] in cand:
  #CE[P]=pt_x
  
  return

pt = cp.asarray([[0,0.5],[1,0.5],[1,1.1]],dtype=cp.int32)
seg_label = cp.asarray([[1,1,1],[1,1,1],[1,1,1]],dtype=cp.int32)
pt_tex = cp.asarray([1,2,1],dtype=cp.int32)
CE = cp.asarray([0, 0, 0],dtype=cp.int32)
THREAD1 = 256 
blocks = ((pt.shape[0]-1) // THREAD1) + 1
pt_to_texture[1, THREAD1](pt, seg_label, pt_tex, CE)

print(CE)