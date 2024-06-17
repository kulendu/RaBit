import sys
import joblib
import pickle
import os
import tqdm
import cv2
import trimesh
import pyrender

import polyscope as ps 
from smplx import SMPL
import numpy as np
import torch
import trimesh
import imageio
from PIL import Image

import argparse

import torch
import torch.nn as nn
import numpy as np
import openmesh as om


'''
TO-DO's:
1. Include the command-line arguments for taking different AMASS files (argparser)
2. Make a separate function for loading the pkl file and the seprate for loading the npz file
3. Refactor the code
'''

import warnings
# the UserWarning can be ignored
warnings.filterwarnings("ignore", category=UserWarning)

# device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = 'cpu'

# polyscope camera parameters
ps.set_up_dir("y_up")
ps.set_front_dir("y_front")
ps.set_navigation_style("free")

# Ensure the output directory exists
output_dir = 'trimesh_frames'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# defining the correspondences
def local_joints():
    pass


ps.init()


class Retargetter:

    def amass_visualizer():

        # amass_file = '/home/kulendu/SMPL-Manipulation/ACCAD/Male2MartialArtsExtended_c3d/Extended_1_stageii.npz'
        pose_file = './AMASS_pose.pkl'
        # amass_file = path_to_amass

        # Load the dataset
        # data = np.load(amass_file)
        data = joblib.load(pose_file)

        print(data.keys())

        frame_count = data['poses'].shape[0]
        # frame_index = 0
        # poses = data['poses'][frame_index]
        betas = data['betas'][:10]
        # trans = data['trans'][frame_index]

        # print(f"Shape Coefficients: {len(betas)}")
        # print(f"Poses: {len(poses)}")

        smpl_model_path = '/home/kulendu/SMPL-Manipulation/SMPL_NEUTRAL.pkl'

        # Load SMPL model
        smpl = SMPL(model_path=smpl_model_path, gender='neutral', create_transl=False)

        # poses = poses[:72]

        '''24*3 = 72 joints : 24 joints and for each joints, 3 DoF
           The first 3 values represents the joint rotation of the root joint (Global orientation)
           The remaining 69 values represent the local rotations of the 23 joints  (Local Orientation) 
        '''

        # Ensure betas have length 10
        if len(betas) != 10:
            raise ValueError(f"Expected shape parameters (betas) of length 10, but got {len(betas)}")

        # Convert pose and shape to tensors with float32 type
        # unsqueeze(0) adds an extra dimension at the 0th position 


        # body_pose = torch.tensor(poses[3:], dtype=torch.float32).unsqueeze(0)  # Exclude the global orientation
        # global_orient = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).unsqueeze(0) # [x,y,z] here the global rotation is dona only along y axis

        # print(f"Original global orientation {global_orient}")

        betas = torch.tensor(betas, dtype=torch.float32).unsqueeze(0)
        # trans = torch.tensor(trans, dtype=torch.float32).unsqueeze(0)
        faces = smpl.faces


        # loop for every frames
        for frame_index in range(frame_count):
            poses = data['poses'][frame_index]
            # betas = data['betas'][:10]
            trans = data['translations'][frame_index]

            print(f"Shape Coefficients: {len(betas)}")
            print(f"Poses: {len(poses)}")

            poses = poses[:72]


            body_pose = torch.tensor(poses[3:], dtype=torch.float32).unsqueeze(0)  # Exclude the global orientation
            global_orient = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).unsqueeze(0) # [x,y,z] here the global rotation is dona only along y axis

            print(f"Original global orientation {global_orient}")


            # Generate body mesh from pose and shape
            output = smpl(global_orient=global_orient, body_pose=body_pose, betas=betas)
            vertices = output.vertices[0].detach().numpy()

            print(f"Global Orientation shape : {global_orient.shape} and length: {len(global_orient)}")
            main_mesh = trimesh.Trimesh(vertices, faces)


            # ps.register_surface_mesh("AMASS_SMPL", vertices, faces)

            # camera_position = [2.0, 2.0, 2.0]  # Change this to your desired camera position
            # look_at_position = [0.0, 0.0, 0.0]  # Change this to your desired target position
            # up_direction = [1.0, 1.0, 0.0]  # Change this to your desired up direction
            
            # ps.look_at((5., 0., 7.), (-0.5, 1., 1.))

            # rabit_data = './rabit_data/shape/mean.obj'

            # mesh = trimesh.load_mesh(main_mesh)
            # vertices = mesh.vertices
            # faces = mesh.faces
            ps_mesh = ps.register_surface_mesh("RaBit", vertices, faces)

            # ps.show()

            # load_rabit()
            
            print("Successfully called the visualizer")
            
            # making directories for storing the video
            filename_per_frame = os.path.join(output_dir, f'frame_{frame_index:04d}.png')
            ps.screenshot(filename_per_frame, transparent_bg=False)
            # ps.clear()
            print(f"Frame: {frame_index}")
            # continue
            break

    # print("------------------- Successfully captured all the frames -------------------")


# loading the rabit model
def load_rabit():
    # mean file of the rabit model
    rabit_data = './rabit_data/shape/mean.obj'

    mesh = trimesh.load_mesh(rabit_data)
    vertices = mesh.vertices
    faces = mesh.faces
    ps_mesh = ps.register_surface_mesh("RaBit", vertices, faces)
    ps.show()

# function for fames compilation
def compile_frames_to_video(output_dir, video_filename, fps=60):
    frame_files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.png')])
    if not frame_files:
        print("No frames to compile")
        return

    # getting the height and width from the first frame
    frame = cv2.imread(frame_files[0])
    print(f"Image Shape: {frame}")
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        video.write(frame)

    video.release()






class Visualizer:
    def trimesh_visualizer():

        amass_file = '/home/kulendu/SMPL-Manipulation/ACCAD/Male2MartialArtsExtended_c3d/Extended_1_stageii.npz'
        # pose_file = './AMASS_pose.pkl'
        # amass_file = path_to_amass

        # Load the dataset
        data = np.load(amass_file)
        # data = joblib.load(amass_file)

        print(f"Data keys for AMASS: {data.keys()}")

        frame_count = data['poses'].shape[0]
        # frame_index = 0
        # poses = data['poses'][frame_index]
        betas = data['betas'][:10]
        # trans = data['trans'][frame_index]

        print(f"Shape Coefficients: {len(betas)}")
        # print(f"Poses: {len(poses)}")

        smpl_model_path = '/home/kulendu/SMPL-Manipulation/SMPL_NEUTRAL.pkl'

        # Load SMPL model
        smpl = SMPL(model_path=smpl_model_path, gender='neutral', create_transl=False)

        # poses = poses[:72]

        '''24*3 = 72 joints : 24 joints and for each joints, 3 DoF
           The first 3 values represents the joint rotation of the root joint (Global orientation)
           The remaining 69 values represent the local rotations of the 23 joints  (Local Orientation) 
        '''

        # Ensure betas have length 10
        if len(betas) != 10:
            raise ValueError(f"Expected shape parameters (betas) of length 10, but got {len(betas)}")

        # Convert pose and shape to tensors with float32 type
        # unsqueeze(0) adds an extra dimension at the 0th position 


        # body_pose = torch.tensor(poses[3:], dtype=torch.float32).unsqueeze(0)  # Exclude the global orientation
        # global_orient = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).unsqueeze(0) # [x,y,z] here the global rotation is dona only along y axis

        # print(f"Original global orientation {global_orient}")

        betas = torch.tensor(betas, dtype=torch.float32).unsqueeze(0)
        # trans = torch.tensor(trans, dtype=torch.float32).unsqueeze(0)
        faces = smpl.faces

        frames = []
        # loop for every frames
        for frame_index in range(frame_count):
            poses = data['poses'][frame_index]
            # betas = data['betas'][:10]
            trans = data['trans'][frame_index]

            print(f"Shape Coefficients: {len(betas)}")
            print(f"Poses: {len(poses)}")

            poses = poses[:72]


            body_pose = torch.tensor(poses[3:], dtype=torch.float32).unsqueeze(0)  # Exclude the global orientation
            global_orient = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).unsqueeze(0) # [x,y,z] here the global rotation is dona only along y axis

            print(f"Original global orientation {global_orient}")


            # Generate body mesh from pose and shape
            output = smpl(global_orient=global_orient, body_pose=body_pose, betas=betas)
            vertices = output.vertices[0].detach().numpy()

            print(f"Global Orientation shape : {global_orient.shape} and length: {len(global_orient)}")
            scene = trimesh.Scene()
            mesh = trimesh.Trimesh(vertices, faces)


            # rabit_data = './rabit_data/shape/mean.obj'

            vertices = mesh.vertices
            faces = mesh.faces
            scene.add_geometry(mesh)
            # mesh.show()
            
            frame_png = scene.save_image(resolution=[1920, 1080], visible=False)
            image_array = np.frombuffer(frame_png, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)

            # Save image using cv2
            cv2.imwrite(os.path.join(output_dir, f'frame_{frame_index:04d}.png'), image)

            # frames.append(imageio.imread(frame_png))




            
            print("Successfully called the visualizer")
            
            # making directories for storing the video
            # filename_per_frame = os.path.join(output_dir, f'frame_{frame_index:04d}.png')
            print(f"Frame: {frame_index}")

    
                    

if __name__ == '__main__':
    # Retargetter.amass_visualizer()
    # compile_frames_to_video(output_dir, 'output_video_from_pkl.mp4', fps=60)
    # Retargetter.load_rabit()
    # Pyrender_visualizer.visualizer()
    Visualizer.trimesh_visualizer()
    compile_frames_to_video(output_dir, 'output_video_from_trimesh.mp4', fps=60)