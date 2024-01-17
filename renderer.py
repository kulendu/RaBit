import os 
import sys
import numpy as np 
from tqdm import tqdm
import polyscope as ps 

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


	def reflect_opengl(self, points):
		points[..., 0]*= -1
		points[..., 1]*= -1

		return points
	


	def render_rabit(self, rabit, SMPL_data, SMPL_model, video_dir=None): 

		T = SMPL_data['joints3d'].shape[0]

		# Get bounding box and object position 
		# SMPL_data['verts'], SMPL_data['joints3d'], _ = SMPL_data.smpl()
		# SMPL_data['verts'] = SMPL_data['verts'].cpu().data.numpy()
		# SMPL_data['joints3d'] = SMPL_data['joints3d'].cpu().data.numpy()

		# # Load 0th frame and get params
		# sample.rabit.load_smpl_params(sample.smpl.smpl_params,0)
		# rabit_verts = sample.rabit.verts
		# rabit_joints = sample.rabit.J


		bbox_smpl = SMPL_data['verts'].max(axis=(0,1))  - SMPL_data['verts'].min(axis=(0,1))
		# bbox_target = rabit_verts.max(axis=0)  - rabit_verts.min(axis=0)

		# bbox = bbox_smpl if np.linalg.norm(bbox_smpl) > np.linalg.norm(bbox_target) else bbox_target
		bbox = bbox_smpl
		object_position = SMPL_data['joints3d'][0,0]

		# camera_position = np.array([0,0,3*self.ps_data['bbox'][0]])
		camera_position = np.array([7*bbox[0],0,0]) + object_position
		look_at_position = np.array([0,0,0]) + object_position
		ps.look_at(camera_position,look_at_position)

		# Translate objects to visualize 
		SMPL_data['joints3d'] += (np.array([0,0,+0.5])*bbox).reshape((1,-1,3))  
		SMPL_data['verts'] += (np.array([0, 0, +0.5]) * bbox).reshape((1,-1,3))

		# rabit_joints += (np.array([0,0,-0.5])*bbox).reshape((1,3))  
		# rabit_verts += (np.array([0, 0, -0.5]) * bbox).reshape((1,3))


		# Initial plot
		ps.remove_all_structures()
		ps_smpl_mesh = ps.register_surface_mesh('SMPL Mesh', self.reflect_opengl(SMPL_data['verts'][0]), SMPL_model.get('f'),transparency=0.5)
		print(' after ps.register')
		SMPL_data["parent_array"] = SMPL_model['kintree_table'][0] 
		# [0,0, 0, 0,1, 2, 3, 4, 5, 6, 7,8,9,9,9,12,13,14,16,17,18,19,20,21] # SMPL Parent Array for bones
		SMPL_data['parent_array'][0] = 0
		print('Kineatic Tree of SMPL', SMPL_model['kintree_table'])


		SMPL_joints_to_show = np.array([8, 12, 9, -1, 13, 10, -1, 14, 11, -1, 20, 23, 1, -1, -1, 0, 5, 2, 6, 3, 7, 4, -1, -1]) # Mapping of the JOINT_NAMES from VIBE <-- SMPL
		print('SMPL Joints -- ', SMPL_joints_to_show)
		SMPL_index_to_show = np.where(SMPL_joints_to_show != -1)[0]
		SMPL_joints_to_show = SMPL_joints_to_show[SMPL_index_to_show]
		print('SMPL joint indexes --- ', SMPL_index_to_show)
		SMPL_index_to_show_dict = dict([(x, i) for i, x in enumerate(SMPL_joints_to_show)])
		print('SMPL_index_to_show_dict -- ', SMPL_index_to_show_dict)
		SMPL_data['parent_array'] = SMPL_data['parent_array'][SMPL_index_to_show]
		print('SMPL_data - parent array', SMPL_data['parent_array'])
		smpl_bone_array = np.array([[SMPL_index_to_show_dict[i], SMPL_index_to_show_dict[p]] for i,p in enumerate(SMPL_data['parent_array']) if i in SMPL_index_to_show_dict and p in SMPL_index_to_show_dict])
		print('SMPL bone array -- ', smpl_bone_array)

		# SMPL_joints_to_show = SMPL_joints_to_show[SMPL_index_to_show]
		
		ps_smpl_skeleton = ps.register_curve_network(f"SMPL Skeleton",self.reflect_opengl(SMPL_data['joints3d'][0, SMPL_joints_to_show]), smpl_bone_array,color=np.array([1,0,0]))

		ps.show()

		# ps_rabit_mesh = ps.register_surface_mesh('RaBit Mesh',rabit_verts,sample.rabit._faces,transparency=0.5)
		
		# rabit_bone_array = np.array([[i,p] for i,p in enumerate(sample.smpl.index['parent_array'])])
		# ps_rabit_skeleton = ps.register_curve_network(f"RaBit Skeleton",rabit_joints,rabit_bone_array,color=np.array([1,0,0]))


		if video_dir is None:
			ps.show()
			return 
		
		os.makedirs(video_dir,exist_ok=True)
		os.makedirs(os.path.join(video_dir,"images"),exist_ok=True)
		os.makedirs(os.path.join(video_dir,"video"),exist_ok=True)

		# ps.show()
		print(f'Rendering images:')
		for i in tqdm(range(SMPL_data['verts'].shape[0])):
			ps_mesh.update_vertex_positions(SMPL_data['verts'][i])
			# ps_target_skeleton.update_node_positions(target_joints[i])
			# ps_smpl_skeleton.update_node_positions(Jtr[i])
			# ps_offset_skeleton.update_node_positions(Jtr_offset[i])
			# ps_joint_mapping.update_node_positions(np.concatenate([target_joints[i,dataset_index],Jtr_offset[i]],axis=0))

			image_path = os.path.join(video_dir,"images",f"smpl_{i}.png")
			# print(f"Saving plot to :{image_path}")	
			ps.set_screenshot_extension(".png");
			ps.screenshot(image_path,transparent_bg=False)
			
			# if i > 0.6*verts.shape[0]:
			# if i  % 100 == 99: 
			# 	ps.show()

		image_path = os.path.join(video_dir,"images",f"smpl_\%d.png")
		video_path = os.path.join(video_dir,"video",f"{test.label}_{test.mcs}_smpl.mp4")
		palette_path = os.path.join(video_dir,"video",f"smpl.png")
		frame_rate = 15
		os.system(f"ffmpeg -y -framerate {frame_rate} -i {image_path} -vf palettegen {palette_path}")
		os.system(f"ffmpeg -y -framerate {frame_rate} -i {image_path} -i {palette_path} -lavfi paletteuse 	 {video_path}")	
		# os.system(f"ffmpeg -y -framerate {frame_rate} -i {image_path} -i {palette_path} -lavfi paletteuse {video_path.replace('mp4','gif')}")	

		print(f"Running Command:",f"ffmpeg -y -framerate {frame_rate} -i {image_path} -i {palette_path} -lavfi paletteuse {video_path}")


