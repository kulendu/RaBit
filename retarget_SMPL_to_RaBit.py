import torch
import torch.nn as nn
import numpy as np
import openmesh as om
from eye_reconstructor import Eye_reconstructor
import sys
import joblib
import pickle

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

    def init_params(self, batch_size):
        
        rabit_params = {}
        rabit_params["theta"] = torch.zeros(batch_size, 72)

        # rabit_params["theta"][:,:3] = torch.from_numpy(np.tile(ROOT_INIT_ROTVEC[None,:],(batch_size,1))) # ROTATION VECTOR to initialize root joint orientation 
        rabit_params["theta"].requires_grad = True

        rabit_params["trans"] = torch.zeros(batch_size, 3)
        rabit_params["trans"].requires_grad = True

        rabit_params["beta"] = torch.ones((1, 100)).to(device)*0.5
        # rabit_params["beta"][0] = 5
        # rabit_params["beta"][1] = 0

        # rabit_params["beta"][self.cfg.TRAIN.MAX_BETA_UPDATE_DIM:] = 0
        rabit_params["beta"].requires_grad = False

        rabit_params["scale"] = torch.ones([1])
        rabit_params["scale"].requires_grad = False

        rabit_params["offset"] = torch.zeros((24,3))
        rabit_params["offset"].requires_grad = False

        print('trans shape ---', rabit_params['trans'].shape)

        self.learning_rate = 1e-3

        for k in rabit_params: 
            rabit_params[k] = nn.Parameter(rabit_params[k].to(device),requires_grad=rabit_params[k].requires_grad)
            self.register_parameter(k,rabit_params[k])
        self.rabit_params = rabit_params

        self.rabit_params['beta'] = self.rabit_params['beta'].repeat(batch_size,1)

        self.optimizer = optim.Adam([{'params': self.rabit_params["scale"], 'lr': self.learning_rate},
                        {'params': self.rabit_params["beta"], 'lr': self.learning_rate},
                        {'params': self.rabit_params["theta"], 'lr': self.learning_rate},{'params': self.rabit_params["trans"], 
                                                                                                         'lr': self.learning_rate},
                        {'params': self.rabit_params["offset"], 'lr': self.learning_rate}])
                         

    def forward(self, beta, pose, trans):
        # NOTE: forward infer

        pose = pose.reshape(pose.shape[0], -1, 3)
        pose = pose[:, self.reorder_index, :]
        trans = trans.unsqueeze(1)  # [1, 1, 3]

        if self.beta_norm:
            # print("minus_mat: ", self.minus_mat.shape, "beta.shape: ", beta.shape, "range_mat:", self.range_mat.shape)
            beta = beta * self.range_mat + self.minus_mat

        if self.theta_norm:
            pose = (pose - 0.5) * 3.1415926
        eye = None      # param of eye is auto

        if eye is not None:
            eye = eye.detach().cpu().numpy()
        return self.update(beta, pose, trans, eye)

    def update(self, beta, pose, trans, eye):
        """
        Called automatically when parameters are updated.

        """
        t_posed = self.update_Tpose_whole(beta)

        # eye reconstruct
        rC, rR = 1.45369, 1.75312
        eyes_list = []
        for i in range(len(t_posed)):
            if eye is not None:
                eyes_mesh = self.eye_recon.reconstruct(t_posed[i].reshape(-1, 3), eye[i, 0], eye[i, 1])
            else:
                eyes_mesh = self.eye_recon.reconstruct(t_posed[i].reshape(-1, 3), rC, rR)
            eyes_list.append(eyes_mesh)

        B = beta.shape[0]
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
            G_loc = self.with_zeros(torch.cat([R[:, i], dJ.reshape(B, 3, 1)], dim=2))
            Gx = torch.matmul(G[int(self.parent[i])], G_loc)
            G.append(Gx)

        G = torch.stack(G, dim=1)

        # remove the transformation due to the rest pose
        zeros24 = torch.zeros((B, 24, 1)).to(device)
        G1 = G - self.pack(
            torch.matmul(
                G,
                torch.cat([J, zeros24], dim=2).reshape([B, 24, 4, 1])
            )
        )

        # transformation of each vertex
        G_r = G1.reshape(B, 24, -1)
        T = torch.matmul(weights, G_r).reshape(B, -1, 4, 4)
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
        # print('shape of posed_vertices -- ', posed_vertices.shape)
        # print('shape of trans ---', trans.shape)
        posed_vertices = posed_vertices + trans


        skeleton = []
        for i in range(len(self.index2cluster)):  # rotate keypoints
            key = self.index2cluster[i]
            if key == 'RootNode':
                skeleton.append(torch.zeros(B, 3).to(device))
                continue
            index_val = posed_vertices[:, self.joint2index[key], :]
            maxval = index_val.max(dim=1)[0]
            minval = index_val.min(dim=1)[0]
            skeleton.append((maxval + minval) / 2)
        for i in range(len(self.additional_7kp_index)):  # toe nose tumb ear
            index_val = posed_vertices[:, self.additional_7kp_index[i], :]
            maxval = index_val.max(dim=1)[0]
            minval = index_val.min(dim=1)[0]
            skeleton.append((maxval + minval) / 2)

        skeleton = torch.stack(skeleton, dim=1)
        #skeleton = joint locations
        #posed_vertices = 3D coord of mesh vertices (SMPL mesh)
        return posed_vertices, skeleton, eyes_list

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

    def update_Tpose_whole(self, beta):
        B = beta.shape[0]
        shapedir = self.shapedirs.to(device)
        v_template = self.v_template.to(device)
        v_shaped = torch.matmul(beta, shapedir.T) + v_template.reshape(1, -1)
        v_posed = v_shaped.reshape(B, -1, 3)
        return v_posed

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


def load_SMPL_data(filepath):
    # Inspect the output file content
    # joints3d : ground truth data
    output = joblib.load(filepath)
    print('Track ids:', output.keys(), end='\n\n')

    pose = output[1]['pose']
    print(output[1].keys())

    return output[1]


def load_SMPL_model():
    with open('./SMPL_NEUTRAL.pkl', 'rb') as smpl_file:
        smpl_model = pickle.load(smpl_file, encoding='latin1')

    print('SMPL keys ---', smpl_model.keys())

    return smpl_model


RaBit_to_SMPL_joint_correspondences = [
 [[0, 12], 0],
 [[1, 0], 1],
 [[2, 3], 2],
 [[3, 9], 3],
 [[4, 16], 4],
 [[5, 18], 5],
 [[6, 18], 6],
 [[7, 20], 7],
 [[8, 22], 8],
 [[10, 15], 10],
 [[11, 1], 11],
 [[12, 4], 12],
 [[13, 2], 13],
 [[14, 5],  14],
 [[15, 7], 15],
 [[16, 10], 16],
 [[17, 17], 17],
 [[19, 19], 19],
 [[20, 21], 20],
 [[21, 23], 21],
 [[22, 8], 22],
 [[23, 11], 23]
]


def train(filepath):
    SMPL_model = load_SMPL_model()
    SMPL_data = load_SMPL_data(filepath)
    SMPL_data['joints3d'] = SMPL_data['joints3d'][:10]
    SMPL_data['verts'] = SMPL_data['verts'][:10]

    joints3d = torch.from_numpy(SMPL_data['joints3d']).to(torch.float32)
    # load some info
    mesh = om.read_polymesh('./rabit_data/shape/mean.obj')
    faces = mesh.face_vertex_indices()

    rabit = RabitModel_eye(beta_norm=True, theta_norm=True)
    
    rabit = RabitModel_eye(beta_norm=True, theta_norm=False)
    rabit.init_params(joints3d.shape[0])

    rabit_joints = []
    SMPL_joints = []
    
    for coresp, index in RaBit_to_SMPL_joint_correspondences:
        rabit_joints.append(coresp[0])
        SMPL_joints.append(coresp[1])


    rabit_joints = np.array(rabit_joints).astype(int)
    SMPL_joints = np.array(SMPL_joints).astype(int)

    # print('RaBit joints --', rabit_joints)
    # print('SMPL joints -- ', SMPL_joints)
    # print('SMPL vertices shape --', SMPL_data['verts'].shape)

    # initial coordinates
    vis = Visualizer()
    body_mesh_points, kps, eyes = rabit(rabit.rabit_params['beta'], rabit.rabit_params['theta'], rabit.rabit_params['trans'])
    body_mesh_points = body_mesh_points.detach().cpu().numpy().reshape(-1, 3)
    # vis.render_rabit(rabit, SMPL_data, SMPL_model, video_dir='demo')
    # print('Shape of Theta---- ', rabit.rabit_params['theta'].shape)
    
    # defining the test data
    for steps in range(50):
        body_mesh_points, kps, eyes = rabit(rabit.rabit_params['beta'], rabit.rabit_params['theta'], rabit.rabit_params['trans'])
        l2_loss = kps[:, rabit_joints, :] - joints3d[:, SMPL_joints, :] # this is a vector
        l2_loss = (l2_loss**2).mean() # converting to scalar
        print('L2 Loss --- ', l2_loss)
        # print('KPS Shape: ', kps[:, rabit_joints, :].shape)
        # print('Joints3D shape: ', joints3d[:, SMPL_joints, :].shape)

        rabit.optimizer.zero_grad() # setting gradients to 0
        l2_loss.backward()
        rabit.optimizer.step() # updating the pose and shape params


        # print('Theta -- after the L2 Loss---', rabit.rabit_params['theta'].grad)
        # print('Trans --  after L2 Loss -- ', rabit.rabit_params['trans'])

        print('-----------Running for ', steps, '-------------') 
        print('Pose Params of Rabit ----- ', rabit.rabit_params['theta'])
        # print('Shape of the root joint: ----- ', rabit.rabit_params['theta'][:, 0,3])
        break

    # print('Joint3D shape', joints3d.shape)
    # print('kps shape', kps.shape)

    
    vis.render_rabit(rabit, SMPL_data, SMPL_model, video_dir='demo')

    mesh = om.PolyMesh(points=body_mesh_points, face_vertex_indices=faces)
    om.write_mesh("output/rabit.obj", mesh)
    om.write_mesh("output/rabit_eyes.obj", eyes[0])
    print("the .obj model with its eyes has been generated")    
    


if __name__ == '__main__':
    filepath =  sys.argv[1] 
    train(filepath)
    
