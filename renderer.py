import os 
import sys
import numpy as np 
from tqdm import tqdm
import polyscope as ps 

import matplotlib.pyplot as plt

# ps.init()
# from retarget_SMPL_to_RaBit import load_SMPL_model
# import trimesh 


class Visualizer: 
	def __init__(self): 
		print('init start')
		ps.init()
		print('init done')

		ps.remove_all_structures()
		# Set camera 
		ps.set_automatically_compute_scene_extents(True)
		ps.set_navigation_style("free")
		# ps.set_view_projection_mode("orthographic")
		# ps.set_ground_plane_mode('shadow_only')


		self.colors = np.array([])

	def reflect_opengl(self, points):
		points[..., 0]*= -1
		points[..., 1]*= -1

		return points
	
	def get_viridis_color(self,vals,min_val=0,max_val=1):
		"""
			get viridis colormap
		"""

		vals = vals.astype(np.float32)
		vals = (vals -min_val)/(max_val - min_val) # Normalize 

		colors = plt.get_cmap('viridis')(vals)

		return colors[:,:3]


	def get_color_from_labels(self,labels):
		num_labels = int(np.unique(labels).max())+1

		if num_labels > self.colors.shape[0]:

			colors = []
			for i in range(num_labels):
				if i < 10:
					colors.append(plt.get_cmap("tab10")(i)[:3])
				elif i < 22:
					colors.append(plt.get_cmap("Paired")(i-10)[:3])
				elif i < 30:
					colors.append(plt.get_cmap("Accent")(i-22)[:3])
				else:
					colors.append(np.random.random(3))

			self.colors = np.array(colors)
		return self.colors[labels,:]

	def get_color_from_matrix(self,weight_matrix):
		# Given a weight matrix showing importance of each weight calculate color as weighted sum 
		num_labels = weight_matrix.shape[1]

		if num_labels > self.colors.shape[0]:

			colors = []
			for i in range(num_labels):
				# if i < 10:
				#     colors.append(plt.get_cmap("tab10")(i)[:3])
				# elif i < 22:
				#     colors.append(plt.get_cmap("Paired")(i-10)[:3])
				# elif i < 30:
				#     colors.append(plt.get_cmap("Accent")(i-22)[:3])
				# else:
				colors.append(np.random.random(3))

			self.colors = np.array(colors)
		return weight_matrix@self.colors[:num_labels,:]
	
	def get_color(self,color):
		if color is None:
			return None
		elif len(np.squeeze(color).shape) == 1:         
			color = color.astype(np.uint64)
			# Get color from a label using matplotlib
			return self.get_color_from_labels(color)
		else: 
			assert color.shape[-1] == 3 or color.shape[-1] == 4, f"Expected information in RGB(Nx3) or RGBA(Nx4) found:{color.shape}" 

			# Normalize colors here for plotting
			if color.max() > 1: color = color.astype(np.float64) / 255
			return color

	def render_rabit(self, rabit_data, SMPL_data, SMPL_model, corresp=None,video_dir=None): 
		"""
			Corresp contains a 2xN np.ndarray 
			which represents the correspondence between the SMPL and RaBit joints
		"""

		

		T = SMPL_data['smpl_joints'].shape[0]

		bbox_smpl = SMPL_data['verts'].max(axis=(0,1))  - SMPL_data['verts'].min(axis=(0,1))
		bbox_rabit = rabit_data['verts'].max(axis=(0,1))  - rabit_data['verts'].min(axis=(0,1))
		# bbox_rabit = bbox_rabit.detach().cpu().numpy()
		# print('Shape of BBox target -', bbox_rabit)

		bbox = bbox_smpl if np.linalg.norm(bbox_smpl) > np.linalg.norm(bbox_rabit) else bbox_rabit
		object_position = SMPL_data['smpl_joints'][0,0]

		# camera_position = np.array([0,0,3*self.ps_data['bbox'][0]])
		camera_position = np.array([0,0,3*bbox[0]]) + object_position
		look_at_position = np.array([0,0,0]) + object_position
		ps.look_at(camera_position,look_at_position)

		# Translate objects to visualize 
		# SMPL_data['smpl_joints'] += (np.array([0,0,+0.5])*bbox).reshape((1,-1,3))  
		# SMPL_data['verts'] += (np.array([0, 0, +0.5]) * bbox).reshape((1,-1,3))

		# rabit_data['joints3d'] += (np.array([0,0,-0.5])*bbox).reshape((1,3))  
		# rabit_data['verts'] += (np.array([0, 0, -0.5]) * bbox).reshape((1,3))


		# SMPL_data['smpl_joints'] = self.reflect_opengl(SMPL_data['smpl_joints'])
		# SMPL_data['verts'] = self.reflect_opengl(SMPL_data['verts'])

		# rabit_data['joints3d'] = self.reflect_opengl(rabit_data['joints3d'])
		# rabit_data['verts'] = self.reflect_opengl(rabit_data['verts'])



		ps.remove_all_structures()

		# Initial plot
		# SMPL Mesh
		ps_smpl_mesh = ps.register_surface_mesh('SMPL Mesh', SMPL_data['verts'][0], SMPL_model.get('f'),transparency=0.5)

		# SMPL Skeleton 
		smpl_bone_array = np.array([ [i,p]  for i,p in enumerate(SMPL_model['parent_array'])])
		ps_smpl_skeleton = ps.register_curve_network(f"SMPL Skeleton",SMPL_data['smpl_joints'][0], smpl_bone_array,color=np.array([0,1,0]))

		# Rabit Mesh 
		ps_rabit_mesh = ps.register_surface_mesh('RaBit Mesh',rabit_data['verts'][0],rabit_data['faces'],transparency=0.5)
		ps_rabit_mesh.add_color_quantity("Skinning Weights", self.get_color_from_labels(rabit_data['color']), enabled=True)

		rabit_bone_array = np.array([[i,p] for i,p in enumerate(rabit_data['parent'])],dtype=np.int64)
		rabit_bone_array[0,1] = 0

		ps_rabit_skeleton = ps.register_curve_network(f"RaBit Skeleton",rabit_data['joints3d'][0],rabit_bone_array,color=np.array([1,0,0]))
		ps_rabit_skeleton_offset = ps.register_curve_network(f"RaBit Skeleton Offset",rabit_data['joints3d_offset'][0],rabit_bone_array,color=np.array([0.75,0.25,0]))


		# Plot correspondence 
		if corresp is not None: 
			print("Corresp:",corresp,corresp.shape)
			corresp_nodes = np.concatenate([ SMPL_data['smpl_joints'][0,corresp[0,:]], rabit_data['joints3d_offset'][0,corresp[1,:]]  ], axis=0)
			corresp_edges = np.array([  (i, i+corresp.shape[1])  for i in range(corresp.shape[1])])

			ps_corresp = ps.register_curve_network("Skeleton Correspodence", corresp_nodes,corresp_edges,radius=0.001)


		if video_dir is None:
			ps.show()
			return 
		



		os.makedirs(video_dir,exist_ok=True)
		os.makedirs(os.path.join(video_dir,"images"),exist_ok=True)
		os.makedirs(os.path.join(video_dir,"video"),exist_ok=True)

		print(f'Rendering images:')
		for i in tqdm(range(T)):
			ps_smpl_mesh.update_vertex_positions(SMPL_data['verts'][i])
			ps_smpl_skeleton.update_node_positions(SMPL_data['smpl_joints'][i])
			ps_rabit_mesh.update_vertex_positions(rabit_data['verts'][i])
			ps_rabit_skeleton.update_node_positions(rabit_data['joints3d'][i])

			image_path = os.path.join(video_dir,"images",f"smpl_{i}.png")
			# print(f"Saving plot to :{image_path}")	
			ps.set_screenshot_extension(".png");
			ps.screenshot(image_path,transparent_bg=False)
			
			# if i > 0.6*verts.shape[0]:
			# if i  % 100 == 99: 
			# 	ps.show()

		image_path = os.path.join(video_dir,"images",f"smpl_\%d.png")
		video_path = os.path.join(video_dir,"video",f"inverse_kinematics.mp4")
		palette_path = os.path.join(video_dir,"video",f"smpl.png")
		frame_rate = 15
		os.system(f"ffmpeg -y -framerate {frame_rate} -i {image_path} -vf palettegen {palette_path}")
		os.system(f"ffmpeg -y -framerate {frame_rate} -i {image_path} -i {palette_path} -lavfi paletteuse 	 {video_path}")	
		# os.system(f"ffmpeg -y -framerate {frame_rate} -i {image_path} -i {palette_path} -lavfi paletteuse {video_path.replace('mp4','gif')}")	

		print(f"Running Command:",f"ffmpeg -y -framerate {frame_rate} -i {image_path} -i {palette_path} -lavfi paletteuse {video_path}")


	def render_shape_iteration(self, rabit_data, SMPL_data, SMPL_model, corresp=None,image_name=None,video_dir=None): 
		"""
			Corresp contains a 2xN np.ndarray 
			which represents the correspondence between the SMPL and RaBit joints
		"""

		

		T = SMPL_data['smpl_joints'].shape[0]

		bbox_smpl = SMPL_data['verts'].max(axis=(0,1))  - SMPL_data['verts'].min(axis=(0,1))
		bbox_rabit = rabit_data['verts'].max(axis=(0,1))  - rabit_data['verts'].min(axis=(0,1))
		# bbox_rabit = bbox_rabit.detach().cpu().numpy()
		# print('Shape of BBox target -', bbox_rabit)

		bbox = bbox_smpl if np.linalg.norm(bbox_smpl) > np.linalg.norm(bbox_rabit) else bbox_rabit
		object_position = SMPL_data['smpl_joints'][0,0]

		# camera_position = np.array([0,0,3*self.ps_data['bbox'][0]])
		camera_position = np.array([0,0,3*bbox[0]]) + object_position
		look_at_position = np.array([0,0,0]) + object_position
		ps.look_at(camera_position,look_at_position)

		# Translate objects to visualize 
		# SMPL_data['smpl_joints'] += (np.array([0,0,+0.5])*bbox).reshape((1,-1,3))  
		# SMPL_data['verts'] += (np.array([0, 0, +0.5]) * bbox).reshape((1,-1,3))

		# rabit_data['joints3d'] += (np.array([0,0,-0.5])*bbox).reshape((1,3))  
		# rabit_data['verts'] += (np.array([0, 0, -0.5]) * bbox).reshape((1,3))


		# SMPL_data['smpl_joints'] = self.reflect_opengl(SMPL_data['smpl_joints'])
		# SMPL_data['verts'] = self.reflect_opengl(SMPL_data['verts'])

		# rabit_data['joints3d'] = self.reflect_opengl(rabit_data['joints3d'])
		# rabit_data['verts'] = self.reflect_opengl(rabit_data['verts'])



		ps.remove_all_structures()

		# Initial plot
		# SMPL Mesh
		ps_smpl_mesh = ps.register_surface_mesh('SMPL Mesh', SMPL_data['verts'][0], SMPL_model.get('f'),transparency=0.5)

		# SMPL Skeleton 
		smpl_bone_array = np.array([ [i,p]  for i,p in enumerate(SMPL_model['parent_array'])])
		ps_smpl_skeleton = ps.register_curve_network(f"SMPL Skeleton",SMPL_data['smpl_joints'][0], smpl_bone_array,color=np.array([0,1,0]))

		# Rabit Mesh 
		ps_rabit_mesh = ps.register_surface_mesh('RaBit Mesh',rabit_data['verts'][0],rabit_data['faces'],transparency=0.5)
		
		rabit_bone_array = np.array([[i,p] for i,p in enumerate(rabit_data['parent'])],dtype=np.int64)
		rabit_bone_array[0,1] = 0

		ps_rabit_skeleton = ps.register_curve_network(f"RaBit Skeleton",rabit_data['joints3d'][0],rabit_bone_array,color=np.array([1,0,0]))
		ps_rabit_skeleton_offset = ps.register_curve_network(f"RaBit Skeleton Offset",rabit_data['joints3d_offset'][0],rabit_bone_array,color=np.array([0.75,0.25,0]))


		# Plot correspondence 
		if corresp is not None: 
			# print("Corresp:",corresp,corresp.shape)
			corresp_nodes = np.concatenate([ SMPL_data['smpl_joints'][0,corresp[0,:]], rabit_data['joints3d_offset'][0,corresp[1,:]]  ], axis=0)
			corresp_edges = np.array([  (i, i+corresp.shape[1])  for i in range(corresp.shape[1])])

			ps_corresp = ps.register_curve_network("Skeleton Correspodence", corresp_nodes,corresp_edges)


		if image_name is not None and video_dir is not None:					
			os.makedirs(video_dir,exist_ok=True)
			os.makedirs(os.path.join(video_dir,"images"),exist_ok=True)
			os.makedirs(os.path.join(video_dir,"video"),exist_ok=True)

			image_path = os.path.join(video_dir,'images',image_name+'.png')
			ps.set_screenshot_extension(".png");
			ps.screenshot(image_path,transparent_bg=False)
		
		else: 
			ps.show()


	def render_shape_iteration_video(self,image_name,video_dir):

		image_path = os.path.join(video_dir,"images",image_name+ '-\%d'+'.png')
		video_path = os.path.join(video_dir,"video",image_name + ".mp4")
		palette_path = os.path.join(video_dir,"video",f"shape_iteration_pallete.png")
		frame_rate = 15
		os.system(f"ffmpeg -y -framerate {frame_rate} -i {image_path} -vf palettegen {palette_path}")
		os.system(f"ffmpeg -y -framerate {frame_rate} -i {image_path} -i {palette_path} -lavfi paletteuse 	 {video_path}")	
		# os.system(f"ffmpeg -y -framerate {frame_rate} -i {image_path} -i {palette_path} -lavfi paletteuse {video_path.replace('mp4','gif')}")	

		print(f"Running Command:",f"ffmpeg -y -framerate {frame_rate} -i {image_path} -i {palette_path} -lavfi paletteuse {video_path}")


