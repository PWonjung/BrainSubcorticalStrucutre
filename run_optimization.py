import multiprocessing
import os
import argparse
import glob
import shutil
import time

label = '17'
def process_input(data_path, sub_id):
    command = f"python ../../voxel2mesh/hippo_p2p.py --data_path {data_path} --tag {label}  --learning_rate 0.0001 --sub_id {sub_id} --gpu 2 --lda 3 0.5 1500 5 1 1 3"
    # pm, cf, edge, lap, norm_con, l2_vert, l2_norm
    
    print(command)
    import subprocess
    subprocess.run(command, shell=True)
def main():
    # time.sleep(2.5*60*60)
    if "synthseg" in label:
        if "l" in label: label_data = "synthseg_l"
        elif "r" in label: label_data = "synthseg_r"
    filelist = glob.glob(f"/root/LV/LV/0124_hippo/{label_data}/*.pkl")
    filelist = filelist[:150]
    print(filelist)
    # print(filelist)python
    num_item = 10
    filelist = filelist
    # print(len(filelist))
    # print(filelist)
    # return
    for i in range(len(filelist)//num_item+1):
        processes = []                                                                                                                                       

        if (i+1)*num_item > len(filelist):
            folders = filelist[i*num_item:]
        else:
            folders = filelist[i*num_item:(i+1)*num_item]
            print(len(folders))
            print(folders)

        for fold in folders:
            sub_id = fold.split("/")[-1].split("_")[0]
            data_path = rf"{fold}"
            exist5000 = glob.glob(f"/root/LV/lv-parametric-modelling/ipynb/edinburgh/edinburgh/0124/{label}/out/{sub_id}/5000*.npy")

            if exist5000==[]:
            # print(f"{sub_id=}")
                p = multiprocessing.Process(target=process_input, args=(data_path, sub_id,))
                processes.append(p)
                p.start()
            else:
                print(f"{sub_id} is already exists.")
    
        for p in processes:
            p.join()


if __name__ == "__main__":
    main()

# python C:\Users\v1wpark\Documents\Projects\lv-parametric-modelling\voxel2mesh\template_redistribute.py --data_path C:\Users\v1wpark\Documents\Projects\lv-parametric-modelling\ipynb\edinburgh\template_mesh_r.pkl --tag template_r  --learning_rate 0.0005 --sub_id template_r
# python ../../voxel2mesh/hippo_p2p.py --data_path /root/LV/LV/0124_hippo/synthseg_l/LBC360002_synthseg_l_opt_data.pkl --tag synthseg_l_new_pm  --learning_rate 0.0005 --sub_id LBC360002 --gpu 1 --lda 0 0.5 1500 5 1 1 1