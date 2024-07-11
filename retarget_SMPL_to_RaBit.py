import os
import sys
import joblib
import pickle
from smplx import SMPL
import trimesh
import argparse

import torch
import torch.nn as nn
import numpy as np
import openmesh as om
from eye_reconstructor import Eye_reconstructor

from renderer import Visualizer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

import warnings
# the UserWarning can be ignored
warnings.filterwarnings("ignore", category=UserWarning)

# device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = 'cpu'


class RabitModel_eye(nn.Module):
    """
    rebuild a RabitModel with eyes.

    Parameters:
    -----------

    """

    def __init__(self, beta_norm=False, theta_norm=False):
        super(RabitModel_eye, self).__init__()
        print("args:\n", "beta_norm:", beta_norm, "theta_norm", theta_norm)
        self.beta_norm = beta_norm
        self.theta_norm = theta_norm
        self.eye_recon = Eye_reconstructor()
        self.prepare()

        self.additional_7kp_index = np.load('./rabit_data/shape/toe_tumb_nose_ear.npy', allow_pickle=True)


        # Hyperparameters
        self.learning_rate = 1e-2
        self.MAX_BETA_UPDATE_DIM = 6 # After 6 dimensions, the tail starts to invert 


    def prepare(self):
        self.dataroot = "./rabit_data/shape/"
        self.mean_file = [self.dataroot + "mean.obj"]
        self.pca_weight = np.load(self.dataroot + "pcamat.npy", allow_pickle=True)[:100, :]
        self.clusterdic = np.load(self.dataroot + 'clusterdic.npy', allow_pickle=True).item()
        self.maxmin = self.processMaxMin()  # [c,r]

        self.index2cluster = {}
        for key in self.clusterdic.keys():
            val = self.clusterdic[key]
            self.index2cluster[val] = key
        self.joint2index = np.load(self.dataroot + 'joint2index.npy', allow_pickle=True).item()
        ktree_table = np.load(self.dataroot + 'ktree_table.npy', allow_pickle=True).item()
        joint_order = np.load("./rabit_data/shape/pose_order.npy")
        self.weightMatrix = np.load(self.dataroot + 'weight_matrix.npy', allow_pickle=True)
        mesh = om.read_polymesh(self.mean_file[0])
        self.points = mesh.points()
        self.cells = mesh.face_vertex_indices()

        # reorder joint
        self.ktree_table = np.ones(24) * -1
        name2index = {}
        for i in range(1, 24):
            self.ktree_table[i] = ktree_table[i][1]
            name2index[ktree_table[i][0]] = i
        reorder_index = np.zeros(24)
        for i, jointname in enumerate(joint_order):
            if jointname in name2index:
                reorder_index[name2index[jointname]] = i
            else:
                reorder_index[0] = 2
        self.reorder_index = np.array(reorder_index).astype(int)

        self.weights = self.weightMatrix
        self.v_template = self.points
        self.shapedirs = self.pca_weight
        self.faces = self.cells
        self.parent = self.ktree_table

        self.pose_shape = [24, 3]
        self.beta_shape = [self.pca_weight.shape[0]]
        self.trans_shape = [3]

        self.shapedirs = torch.from_numpy(self.shapedirs).T.to(torch.float32)
        self.v_template = torch.from_numpy(self.v_template).to(torch.float32)
        self.weights = torch.from_numpy(self.weightMatrix).to(torch.float32)


    def init_params(self, batch_size):
        
        rabit_params = {}
        rabit_params["theta"] = torch.zeros(batch_size, 72)
        # rabit_params["theta"][:,:3] = torch.from_numpy(np.tile(ROOT_INIT_ROTVEC[None,:],(batch_size,1))) # ROTATION VECTOR to initialize root joint orientation 
        rabit_params["theta"].requires_grad = False

        rabit_params["trans"] = torch.zeros(batch_size, 3)
        rabit_params["trans"].requires_grad = True

        print("Using device:",device,self.shapedirs.shape)

        rabit_params["beta"] = torch.ones(100).to(device)*0.5


        # rabit_params["beta"][self.MAX_BETA_UPDATE_DIM:] = 0
        rabit_params["beta"].requires_grad = True

        rabit_params["scale"] = 2*torch.ones([1])
        # rabit_params["scale"] = torch.ones([1])
        rabit_params["scale"].requires_grad = True

        rabit_params["offset"] = torch.zeros((24,3))
        rabit_params["offset"].requires_grad = False



        for k in rabit_params: 
            rabit_params[k] = nn.Parameter(rabit_params[k].to(device),requires_grad=rabit_params[k].requires_grad)
            self.register_parameter(k,rabit_params[k])
        self.rabit_params = rabit_params



        self.optimizer = optim.Adam([{'params': self.rabit_params["scale"], 'lr': self.learning_rate},
                        {'params': self.rabit_params["beta"], 'lr': self.learning_rate},
                        {'params': self.rabit_params["theta"], 'lr': self.learning_rate},
                        {'params': self.rabit_params["trans"],'lr': self.learning_rate},
                        {'params': self.rabit_params["offset"], 'lr': self.learning_rate}])
        
        # optimzer for the pose loop
        self.optimizer2 = optim.Adam([{'params': self.rabit_params["scale"], 'lr': self.learning_rate},
                        {'params': self.rabit_params["beta"], 'lr': 0},
                        {'params': self.rabit_params["theta"], 'lr': self.learning_rate},
                        {'params': self.rabit_params["trans"],'lr': self.learning_rate},
                        {'params': self.rabit_params["offset"], 'lr': self.learning_rate}])
                         
                         

    def forward(self, beta, pose, trans):
        # NOTE: forward infer

        pose = pose.reshape(pose.shape[0], -1, 3)
        # pose = pose[:, self.reorder_index, :]
        trans = trans.unsqueeze(1)  # [1, 1, 3]

        if self.beta_norm:
            # print("minus_mat: ", self.minus_mat.shape, "beta.shape: ", beta.shape, "range_mat:", self.range_mat.shape)
            beta = beta * self.range_mat + self.minus_mat

        beta = beta.reshape((1,-1)).repeat(pose.shape[0],1) # Duplicate beta for each timestep


        if self.theta_norm:
            pose = (pose - 0.5) * np.pi
        eye = None      # param of eye is auto

        if eye is not None:
            eye = eye.detach().cpu().numpy()
        return self.update(beta, pose, trans, eye)

    def update(self, beta, pose, trans, eye):
        """
        Called automatically when parameters are updated.

        """
        t_posed = self.update_Tpose_whole(beta)

        eyes_list = self.update_Tpose_eyes(eye,t_posed) # Get the location of eyes in the T-pose

        B = beta.shape[0] # Batch size
        weights = self.weights.to(device)

        J = []
        for i in range(len(self.index2cluster)):
            key = self.index2cluster[i]
            if key == 'RootNode':
                J.append(torch.zeros(B, 3).to(device))
                continue
            index_val = t_posed[:, self.joint2index[key], :]
            maxval = index_val.max(dim=1)[0]
            minval = index_val.min(dim=1)[0]

            J.append((maxval + minval) / 2)
        J = torch.stack(J, dim=1)
        
        # rotation matrix for each joint
        R = self.rodrigues(pose)

        # world transformation of each joint
        G = []
        G.append(self.with_zeros(torch.cat([R[:, 0], J[:, 0, :].reshape(B, 3, 1)], dim=2)))
        for i in range(1, self.ktree_table.shape[0]):
            dJ = J[:, i, :] - J[:, int(self.parent[i]), :] 
            # dJ = J[:, i, :] - J[:, int(self.parent[i]), :] 
            G_loc = self.with_zeros(torch.cat([R[:, i], dJ.reshape(B, 3, 1)], dim=2))
            Gx = torch.matmul(G[int(self.parent[i])], G_loc)
            G.append(Gx)

        G = torch.stack(G, dim=1)

        # remove the transformation due to the rest pose (not sure why this is being done but otherwise gives a degenrate result)
        zeros24 = torch.zeros((B, 24, 1)).to(device)
        G1 = G - self.pack(
            torch.matmul(
                G,
                torch.cat([J, zeros24], dim=2).reshape([B, 24, 4, 1])
            )
        )

        # print(f"G:{G.shape}, G1:{G1.shape} Weights:{weights.shape}")    
        # transformation of each vertex
        G_r = G1.reshape(B, 24, -1)
        T = torch.matmul(weights, G_r).reshape(B, -1, 4, 4) # BxVx16 -> BxVx4x4
        ones_vposed = torch.ones((B, t_posed.shape[1], 1)).to(device)
        rest_shape_h = torch.cat([t_posed, ones_vposed], dim=2).reshape(B, -1, 4, 1)

        tempmesh = eyes_list[0]
        faces = tempmesh.face_vertex_indices()
        for i in range(len(T)):
            eye_mesh = eyes_list[i]
            eye_points = eye_mesh.points()
            Ti = T[i, self.eye_recon.eyeidx]
            Ti = Ti.mean(0).detach().cpu().numpy()  # 4*4
            temp = np.ones(len(eye_points))
            eye_points = np.c_[eye_points, temp]
            eye_points = eye_points.dot(Ti.T)[:, :3]
            eye_mesh = om.PolyMesh(points=eye_points, face_vertex_indices=faces)
            eyes_list[i] = eye_mesh

         
        posed_vertices = torch.matmul(T, rest_shape_h).reshape(B, -1, 4)[:, :, :3]
        
        # Scale vertices
        posed_vertices_center = posed_vertices.mean(dim=1,keepdims=True)
        posed_vertices = self.rabit_params["scale"]*(posed_vertices - posed_vertices_center) + posed_vertices_center
        
        # print('shape of posed_vertices -- ', posed_vertices.shape)
        # print('shape of trans ---', trans.shape)
        posed_vertices = posed_vertices + trans


        skeleton = []
        
        # Not sure why RaBit uses this ? Basically they take the
        # for i in range(len(self.index2cluster)):  # rotate keypoints
        #     key = self.index2cluster[i]
        #     if key == 'RootNode':
        #         skeleton.append(torch.zeros(B, 3).to(device))
        #         continue
        #     index_val = posed_vertices[:, self.joint2index[key], :]
        #     maxval = index_val.max(dim=1)[0]
        #     minval = index_val.min(dim=1)[0]
        #     skeleton.append((maxval + minval) / 2)

        skeleton = self.rabit_params['scale']*(G[:,:,:3,3] - posed_vertices_center) + posed_vertices_center + trans

        additional_keypoints = []
        for i in range(len(self.additional_7kp_index)):  # toe nose tumb ear
            index_val = posed_vertices[:, self.additional_7kp_index[i], :]
            maxval = index_val.max(dim=1)[0]
            minval = index_val.min(dim=1)[0]
            additional_keypoints.append((maxval + minval) / 2)
        additional_keypoints = torch.stack(additional_keypoints,dim=1)

        skeleton = torch.cat([skeleton,additional_keypoints], dim=1)

        # For putting additional offset loss similar to soft-margin svm 
        skeleton_offset = J + self.rabit_params['offset'] # Adding offset
        skeleton_offset_homo = torch.cat([skeleton_offset,torch.ones((B,24,1)).to(device)],2) # Converting to homogenous co-ordinates 
        skeleton_offset_transformed = (G1*skeleton_offset_homo.unsqueeze(2)).sum(3) # Use the relative transformation matrix to deform each joint   
        skeleton_offset_transformed = skeleton_offset_transformed[:,:,:3] # Back to 3D co-ordinates
        skeleton_offset_scaled = self.rabit_params['scale']*(skeleton_offset_transformed - posed_vertices_center) + posed_vertices_center + trans 

        return posed_vertices, skeleton,skeleton_offset_scaled, eyes_list


    def update_Tpose_whole(self, beta):
        B = beta.shape[0]
        shapedir = self.shapedirs.to(device)
        v_template = self.v_template.to(device)
        v_shaped = torch.matmul(beta, shapedir.T) + v_template.reshape(1, -1)
        v_posed = v_shaped.reshape(B, -1, 3)
        return v_posed

    def update_Tpose_eyes(self,eye,t_posed): 
        # eye reconstruct
        rC, rR = 1.45369, 1.75312
        eyes_list = []
        for i in range(len(t_posed)):
            if eye is not None:
                eyes_mesh = self.eye_recon.reconstruct(t_posed[i].reshape(-1, 3), eye[i, 0], eye[i, 1])
            else:
                eyes_mesh = self.eye_recon.reconstruct(t_posed[i].reshape(-1, 3), rC, rR)
            eyes_list.append(eyes_mesh)
        
        return eyes_list

    def rodrigues(self, r):
        # r shape B (24, 3)
        B = r.shape[0]
        theta = torch.norm(r, p=2, dim=2, keepdim=True)
        theta = torch.clip(theta, min=1e-6)  # avoid zero divide
        r_hat = r / theta
        z_stick = torch.zeros((B, theta.shape[1], 1)).to(device)

        m = torch.cat([
            z_stick, -r_hat[:, :, 2:3], r_hat[:, :, 1:2],
            r_hat[:, :, 2:3], z_stick, -r_hat[:, :, 0:1],
            -r_hat[:, :, 1:2], r_hat[:, :, 0:1], z_stick], dim=2)
        m = m.reshape(B, -1, 3, 3)

        i_cube = [torch.eye(3).unsqueeze(0) for i in range(theta.shape[1])]
        i_cube = torch.cat(i_cube, dim=0).to(device)

        r_hat = r_hat.unsqueeze(3)
        r_hat_T = r_hat.transpose(3, 2)
        r_hat_M = torch.matmul(r_hat, r_hat_T)

        cos = torch.cos(theta).unsqueeze(2)
        sin = torch.sin(theta).unsqueeze(2)

        R = cos * i_cube + (1 - cos) * r_hat_M + sin * m
        return R

    def with_zeros(self, x):
        B = x.shape[0]
        constant1 = torch.zeros((B, 1, 3))
        constant2 = torch.ones((B, 1, 1))
        constant = torch.cat([constant1, constant2], dim=2).to(device)
        return torch.cat([x, constant], dim=1)

    def pack(self, x):
        B = x.shape[0]
        t1 = torch.zeros((B, x.shape[1], 4, 3)).to(device)
        return torch.cat([t1, x], dim=3)

    def processMaxMin(self):
        maxmin = np.load(self.dataroot + 'maxmin.npy', allow_pickle=True)
        maxmin = maxmin.T
        maxmin = maxmin[:100, [1, 0]]
        c = maxmin[:, 0:1]
        norm_maxmin = maxmin - c
        r = norm_maxmin[:, 1:]
        c = c.reshape(-1)
        r = r.reshape(-1)
        c = torch.from_numpy(c).to(torch.float32).to(device)
        r = torch.from_numpy(r).to(torch.float32).to(device)
        self.minus_mat = c
        self.range_mat = r
        return




def get_local_joints(data,model): 
    # Get 24x3 SMPl joints using mean of vertices
    # Refer: https://github.com/mkocabas/VIBE/issues/18
    # verts = data['verts']
    # J_regressor = model['J_regressor']
    verts = SMPL2RabitRetargetter.smpl_model['vertex_ids']
    J_regressor = SMPL2RabitRetargetter.smpl_model['J_regressor']
    # verts = SMPL2RabitRetargetter.smpl_model['ver']
    local_joints = np.einsum('jv,tvd->tjd',J_regressor.toarray(),verts)
    return local_joints


RaBit_to_SMPL_joint_correspondences = [
 [9,12],
 [1,0],
#  [2,6],
 [3,9],
#  [4,13], # Left collarbone
 [6,18],
 [7,20],
 [8,22],
#  [10,15], # Neck collarbone
 [11,1],
 [12,4],
 [13,2],
 [14,5],
 [15,7],
 [16,10],
#  [17,14], # Right collarbone
 [19,19],
 [20,21],
 [21,23],
 [22,8],
 [23,11]
]

RaBit_to_SMPL_joint_correspondences = np.array(RaBit_to_SMPL_joint_correspondences)
corresp = np.zeros_like(RaBit_to_SMPL_joint_correspondences)

# Reversing the mapping: For mapping from SMPL to RaBbit, instead of RaBit to SMPL
corresp[:,0] = RaBit_to_SMPL_joint_correspondences[:,1]
corresp[:,1] = RaBit_to_SMPL_joint_correspondences[:,0]
corresp = corresp.T # Transpose to match the shape

class SMPL2RabitRetargetter: 
    def __init__(self,args):


        self.args = args
        self.smpl = self.load_smpl(args.smpl)
        self.data = self.load_VIBE_data(args.file)
        # self.data = self.load_AMASS_data(args.amass)


        self.data['verts'] = np.tile(self.smpl['v_template'].reshape((1,-1,3)) , (self.data['verts'].shape[0],1,1) )

        if args.debug: 
            self.data['joints3d'] = self.data['joints3d'][:2]
            self.data['verts'] = self.data['verts'][:2]
            self.data['pose'] = self.data['pose'][:2]
        else: 
            self.data['joints3d'] = self.data['joints3d'][::5]
            self.data['verts'] = self.data['verts'][::5]
            self.data['pose'] = self.data['pose'][::5]

        self.data['smpl_joints'] = get_local_joints(self.data,self.smpl) # Use SMPL J-regressor to get smpl joints (root retative)    
        self.smpl['parent_array'] = self.smpl['kintree_table'][0]
        self.smpl['parent_array'][0] = 0

        
        # load RaBit module 
        self.rabit = RabitModel_eye(beta_norm=True, theta_norm=False)
        self.rabit.init_params(self.data['verts'].shape[0])

        self.vis = Visualizer()

    @staticmethod
    def load_VIBE_data(filepath):
        # Inspect the output file content
        # joints3d : ground truth data
        output = joblib.load(filepath)
        print('Track ids:', output.keys(), end='\n\n')

        pose = output[1]['pose']
        print(output[1].keys())

        
        return output[1]


    @staticmethod
    def load_AMASS_data(filepath):
        thetas = joblib.load(filepath)
        print('Thetas from AMASS:', thetas.keys())


    @staticmethod
    def load_smpl(smpl_body_model_path):
        with open(smpl_body_model_path, 'rb') as smpl_file:
            smpl_model = pickle.load(smpl_file, encoding='latin1')

        print('SMPL keys ---', smpl_model.keys())

        return smpl_model

    def stage_1_shape_parameters_matching(self):

        # rabit_joints = []
        # SMPL_joints = []
        # for coresp in RaBit_to_SMPL_joint_correspondences:
        #     print(coresp)
        #     rabit_joints.append(coresp[0])
        #     SMPL_joints.append(coresp[1])
        # corresp = np.array([SMPL_joints,rabit_joints])    

        rabit_joints = RaBit_to_SMPL_joint_correspondences[:,0].astype(int)
        SMPL_joints = RaBit_to_SMPL_joint_correspondences[:,1].astype(int)

        # initial coordinates
        self.visualize(video_dir=None)        


        # Target joints: from np array to torch arrays
        joints3d = torch.from_numpy(self.data['smpl_joints']).to(torch.float32)
        joints3d = joints3d.to(device)

        # Stage:1 Shape matching (neutral pose is matching is done. )
        for step in range(100):
            body_mesh_points, kps,kps_offset, eyes = self.rabit(self.rabit.rabit_params['beta'], self.rabit.rabit_params['theta'], self.rabit.rabit_params['trans'])
            l2_loss = kps_offset[:, rabit_joints, :] - joints3d[:, SMPL_joints, :] # element-wise differnce: this is a vector
            l2_loss = (l2_loss**2).mean() # converting to scalar
            '''l2_loss = (RaBit joints - SMPL joints)**2'''
            
            loss_offset_min = self.rabit.rabit_params['offset'].norm() # Add offset between the joint positions of RaBit and SMPL. Not useful
            loss_beta_norm = (self.rabit.rabit_params['beta'] - 0.5).norm() # Force Beta values to be near their mean. Loss value above 0.5 gives degenerate results. Around 0.23 is fine. 


            self.rabit.optimizer.zero_grad() # setting gradients to 0
            
            loss = l2_loss
            loss += 0.01*loss_beta_norm # 0.01 is perfect givens 0.23 beta loss term. 0.001 is too low; 0.1 is too high  
            
            # loss = l2_loss + 0.01*loss_beta_norm + 1*loss_offset_min

            # loss = l2_loss +  0.005*loss_offset_min
            # loss = l2_loss


            loss.backward()
            print(f'Iteration: {step} Loss ---  Total:{loss.data.item()} L2:{l2_loss.data.item()} Offset: {loss_offset_min.data.item()} Beta:{loss_beta_norm.data.item()}')

            # Update which beta parameters will be updated
            # if the gradients of the betas is not None, then set the gradients beyond a certain limit (MAX_BETA_UPDATE_DIM) to 0
            if self.rabit.rabit_params['beta'].grad is not None:
                self.rabit.rabit_params['beta'].grad[self.rabit.MAX_BETA_UPDATE_DIM:] = 0 


            self.rabit.optimizer.step() # updating the pose and shape params




            rabit_data = {'verts': body_mesh_points.detach().cpu().numpy(), 
                      'joints3d': kps.detach().cpu().numpy(),
                      'joints3d_offset': kps_offset.detach().cpu().numpy(),
                      'parent': self.rabit.parent,
                       'faces': self.rabit.faces }
            
            # self.vis.render_shape_iteration(rabit_data, self.data, self.smpl, corresp = corresp, image_name=f"shape_iteration-{step}",video_dir="demo")
        
        # self.vis.render_shape_iteration_video(image_name=f"shape_iteration",video_dir="demo")

        print('Offset:', self.rabit.rabit_params['offset'][0])
        print('Trans:', self.rabit.rabit_params['trans'][0])
        print(f'Source Root Location:',kps_offset[0,0])
        print('Target Root Trans:', joints3d[0,12])
        print('Scale:', self.rabit.rabit_params['scale'])
        print('Beta:', self.rabit.rabit_params['beta'][:self.rabit.MAX_BETA_UPDATE_DIM])
            

        self.visualize(video_dir=None)
<<<<<<< HEAD

    # loading the vertices from the SMPL model 
    '''
    TO-DOs:
    1. Pass the poses of the VIBE to the SMPL model class
    2. In the get_local_joints function replace the joints and the vertices
    '''
    smpl_path = './SMPL_NEUTRAL.pkl'
    smpl_model = SMPL(model_path=smpl_path, gender='neutral', create_transl=False) # deforming mesh from the SMPL model - smplx
    
    # defining the vertices and the faces
    # vertices = trimesh
    # faces = 

=======
        
        # Save rabit parameters from stage-1
        torch.save(self.rabit.state_dict(), self.args.save)
     
>>>>>>> 3cd4f3fd0ca512bf79711fa5a5485ace07083409
        
    # retargetter module
    def stage_2_pose_parameters_matching(self):

        # self.visualize(video_dir=None)       

        # Load previous saved resutls from stage-1
        if os.path.isfile(self.args.save):
            try:
                self.rabit.load_state_dict(torch.load(self.args.save))
            except Exception as e:
                print("Unable to load recomputed rabit parameters. Error:", e)


        rabit_joints = RaBit_to_SMPL_joint_correspondences[:,0].astype(int)
        SMPL_joints = RaBit_to_SMPL_joint_correspondences[:,1].astype(int)


        # Load VIBE/AMASS joint locations
        self.data = self.load_VIBE_data(self.args.file)

        # Convert to from opengl format 
        self.data['verts'][...,[1,2]] *= -1



        if self.args.debug: 
            self.data['joints3d'] = self.data['joints3d'][:2]
            self.data['verts'] = self.data['verts'][:2]
            self.data['pose'] = self.data['pose'][:2]
        else:
            self.data['joints3d'] = self.data['joints3d'][::5]
            self.data['verts'] = self.data['verts'][::5]
            self.data['pose'] = self.data['pose'][::5]

        self.data['smpl_joints'] = get_local_joints(self.data,self.smpl) # Use SMPL J-regressor to get smpl joints (root retative)


        self.visualize(video_dir=None)       

        # Target joints: from np array to torch arrays
        joints3d = torch.from_numpy(self.data['smpl_joints']).to(torch.float32)
        joints3d = joints3d.to(device)

        self.rabit.rabit_params["theta"].requires_grad = True
        self.rabit.rabit_params["beta"].requires_grad = True
        self.rabit.rabit_params["trans"].requires_grad = True
        self.rabit.rabit_params["offset"].requires_grad = False # Don't update offset. Not positive outcome

        self.learning_rate = 0.1

        self.optimizer = optim.Adam([
                                     {'params': self.rabit.rabit_params["scale"], 'lr': self.learning_rate},
                                     {'params': self.rabit.rabit_params["theta"], 'lr': self.learning_rate},
                                     {'params': self.rabit.rabit_params["trans"],'lr': self.learning_rate},
                                     {'params': self.rabit.rabit_params["beta"], 'lr': 0.1*self.learning_rate}, # Updating too fast. Reducing speed to give for slower update
                                     {'params': self.rabit.rabit_params["offset"], 'lr': self.learning_rate}, # Updating too fast. Reducing speed to give for slower update
                                     ])



        for step in range(200):
            # self.rabit_params["theta"].requires_grad = True
            # self.rabit_params["beta"].requires_grad = False

            print(self.rabit.rabit_params['beta'][:self.rabit.MAX_BETA_UPDATE_DIM])

            # self.rabit.rabit_params['']
            body_mesh_points, kps,kps_offset, eyes = self.rabit(self.rabit.rabit_params['beta'], self.rabit.rabit_params['theta'], self.rabit.rabit_params['trans'])
            l2_loss = kps_offset[:, rabit_joints, :] - joints3d[:, SMPL_joints, :] # element-wise differnce: this is a vector
            l2_loss = (l2_loss**2).mean() # converting to scalar
            '''l2_loss = (RaBit joints - SMPL joints)**2'''
            
            loss_offset_min = self.rabit.rabit_params['offset'].norm() # Add offset between the joint positions of RaBit and SMPL. Not useful
            loss_theta_norm = (self.rabit.rabit_params['theta']**2).mean() # Force theta values to be near their mean. Not useful 
            loss_beta_norm = (self.rabit.rabit_params['beta'] - 0.5).norm()**2 # Force Beta values to be near their mean. Not useful

<<<<<<< HEAD
            self.rabit.optimizer2.zero_grad() # setting gradients to 0
=======
            self.optimizer.zero_grad() # setting gradients to 0
>>>>>>> 3cd4f3fd0ca512bf79711fa5a5485ace07083409
            
            # loss = l2_loss + 0.01*loss_beta_norm + 1*loss_offset_min
            # loss = l2_loss +  0.005*loss_offset_min
            loss = l2_loss
            # loss += 0.0001*loss_beta_norm 
            loss += 0.01*loss_theta_norm # Very small loss on theta to prevent degenreate poses. Less than 
            # loss += 0.1*loss_offset_min # Very small loss on theta to prevent degenreate poses. Less than 


            loss.backward()
<<<<<<< HEAD
            # self.rabit.rabit_params["beta"].grad[:]=0
            print(f'Iteration: {step} Loss ---  Total:{loss.data.item()} L2:{l2_loss.data.item()} Offset: {loss_offset_min.data.item()} Theta:{loss_theta_norm.data.item()} Betas:{loss_theta_norm.data.item()}')
=======
            print(f'Iteration: {step} Loss ---  Total:{loss.data.item()} L2:{l2_loss.data.item()} Offset: {loss_offset_min.data.item()} Theta:{loss_theta_norm.data.item()} Betas:{loss_beta_norm.data.item()}')
>>>>>>> 3cd4f3fd0ca512bf79711fa5a5485ace07083409

            # Update which beta parameters will be updated
            # if the gradients of the betas is not None, then set the gradients beyond a certain limit (MAX_BETA_UPDATE_DIM) to 0
            if self.rabit.rabit_params['beta'].grad is not None:
                self.rabit.rabit_params['beta'].grad[self.rabit.MAX_BETA_UPDATE_DIM:] = 0 


<<<<<<< HEAD
            self.rabit.optimizer2.step() # updating the pose and shape params
=======
            self.optimizer.step() # updating the pose and shape params
>>>>>>> 3cd4f3fd0ca512bf79711fa5a5485ace07083409




            rabit_data = {'verts': body_mesh_points.detach().cpu().numpy(), 
                      'joints3d': kps.detach().cpu().numpy(),
                      'joints3d_offset': kps_offset.detach().cpu().numpy(),
                      'parent': self.rabit.parent,
                       'faces': self.rabit.faces }
            
            

            
        
            self.vis.render_shape_iteration(rabit_data, self.data, self.smpl, corresp = corresp, image_name=f"pose_iteration-{step}",video_dir="demo")
        


        self.vis.render_shape_iteration_video(image_name=f"pose_iteration",video_dir="demo")



        # self.visualize(video_dir=None)
        

        torch.save(self.rabit.state_dict(), self.args.save)
        
        self.rabit.rabit_params['trans'].data[:,0] += self.rabit.rabit_params['scale']*0.5    
        self.rabit.rabit_params['beta'].data[:] = 0.5

        self.visualize(video_dir="demo2")
        




        # Init root pose
        # with torch.no_grad():
        #     self.rabit.rabit_params['theta'][:,:3] = torch.from_numpy(self.data['pose'][:,:3])     
        # print('Theta -- after the L2 Loss---', self.rabit.rabit_params['theta'])
        # self.visualize(video_dir='demo')
        # pass 


    def retarget(self): 
        # self.stage_1_shape_parameters_matching()
        self.stage_2_pose_parameters_matching()

    def visualize(self,video_dir=None):    
        body_mesh_points, kps,kps_offset, eyes = self.rabit(self.rabit.rabit_params['beta'], self.rabit.rabit_params['theta'], self.rabit.rabit_params['trans'])
        
        rabit_data = {'verts': body_mesh_points.detach().cpu().numpy(), 
                      'joints3d': kps.detach().cpu().numpy(),
                      'joints3d_offset': kps_offset.detach().cpu().numpy(),
                      'parent': self.rabit.parent,
                        'faces': self.rabit.faces,
                        'color': np.argmax(self.rabit.weightMatrix,axis=1) }

        self.vis.render_rabit(rabit_data, self.data, self.smpl, corresp=corresp, video_dir=video_dir)

        
if __name__ == '__main__':


    ############################# Command line Argument Parser #######################################################
    parser = argparse.ArgumentParser(
                        prog='Retargetting',
                        description='Retargets from SMPL to RaBit',
                        epilog='')
    parser.add_argument('--file',type=str, default="../data/SMPL/vibe_output.pkl", help="Path to .trc file that needs to be retargeted.")  # path to trc file
    parser.add_argument('--save',type=str, default="../data/SMPL/retarget.pt", help="Path to .trc file that needs to be retargeted.")  # path to trc file
    parser.add_argument('--amass',type=str, default="./AMASS_pose.pkl", help="Path to the pkl file for AMASS data.")  # path to trc file
    parser.add_argument('--smpl', default="SMPL_NEUTRAL.pkl", help="Path to .pkl SMPL model used for pose estimation")
    parser.add_argument('-f', '--force',action='store_true',help="forces a re-run on retargetting even if pkl file containg smpl data is already present.")  # on/off flag
    parser.add_argument('--render', action='store_true', help="Render a video and save it it in RENDER_DIR. Can also be set in the utils.py")  # on/off flag
    parser.add_argument('--debug', dest='debug', action='store_true', default=True,  help="Debug and run on less number of frames")
    parser.add_argument('--no-debug', dest='debug', action='store_false', help="Debug and run on less number of frames")
    parser.add_argument('--gpu', dest='gpu', action='store_true', default=True,help="Whether to use CUDA or CPU")
    parser.add_argument('--no-gpu', dest='gpu', action='store_false',help="Whether to use CUDA or CPU")

    cmd_line_args = parser.parse_args()

    print("GPU:",cmd_line_args.gpu)
    print("Debug:",cmd_line_args.debug)

    if cmd_line_args.gpu: 
        device = 'cuda' 

    retargetter = SMPL2RabitRetargetter(cmd_line_args).retarget()
    
