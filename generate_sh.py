import os
import json
import cv2
import numpy as np

def read_cam_params(cam_path):
    with open(cam_path) as f:
        cam_params = json.load(f)
        for key1 in cam_params:
            for key2 in cam_params[key1]:
                cam_params[key1][key2] = np.array(cam_params[key1][key2])
    return cam_params

def render(self, vertices, frame, cam_params, vertices_in_world=True):
    blending_weight = 1.0
    if vertices_in_world:
        vertices = np.matmul(vertices - cam_params['extrinsics']['T'], np.transpose(cam_params['extrinsics']['R']))

    vertices_to_render = vertices
    intrinsics = cam_params['intrinsics_wo_distortion']['f'].tolist() + cam_params['intrinsics_wo_distortion'][
        'c'].tolist()
    background_image = frame

    vertex_colors = np.ones([vertices_to_render.shape[0], 4]) * [0.9, 0.9, 0.9, 1]
    tri_mesh = trimesh.Trimesh(vertices_to_render, self.smplx_model.faces,
                                vertex_colors=vertex_colors)

    mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
    # scene = pyrender.Scene(ambient_light=(0.0, 0.0, 0.0))
    scene.add(mesh, 'mesh')

    camera_pose = np.eye(4)
    rot = trimesh.transformations.euler_matrix(0, np.pi, np.pi, 'rxyz')
    camera_pose[:3, :3] = rot[:3, :3]

    camera = pyrender.IntrinsicsCamera(
        fx=intrinsics[0],
        fy=intrinsics[1],
        cx=intrinsics[2],
        cy=intrinsics[3])

    scene.add(camera, pose=camera_pose)

    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=10)

    scene.add(light, pose=camera_pose)
    color, rend_depth = self.m_renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    # img = color.astype(np.float32) / 255.0

    # blended_image = img[:, :, :3]

    valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
    blended_image = color[:, :, :-1] * valid_mask + (1 - valid_mask) * background_image
    # color = output_img.astype(np.uint8)

    return blended_image

def smplxRenderCam(frame_idx=None):
    rendered_image = render(smpl_model.vertices[frame_idx], frame_img, cam_params,
                                            vertices_in_world=False)

    save_fn = '%s_%s_%s_%04d'%(self.subset,self.seq_name,self.action_name,frame_idx)
    # out_mesh = trimesh.Trimesh(vertices=camera_posed_data.vertices[frame_idx],faces=self.smplx_model.faces)
    # out_mesh.export(osp.join(self.debug_dir,save_fn+'.obj'))
    img = np.concatenate([frame_img,rendered_image],axis=1)
    cv2.imwrite(osp.join(self.debug_dir,save_fn+'.jpg'),img[:,:,::-1])

def generate_sh(input_path, sh_file):
    recording_path = os.path.join(input_path, 'recordings')
    camera_path = os.path.join(input_path, 'camera')
    dirs = os.listdir(recording_path)
    
    # 追加写入 sh 脚本,批量处理
    for recording_dir in dirs:
        recording_dir_path = os.path.join(recording_path, recording_dir)
        cam_params_path = os.path.join(camera_path, recording_dir, recording_dir + '.json')
        cam_params = read_cam_params(cam_params_path)
        
        focal_length_x = float(cam_params['intrinsics_wo_distortion']['f'][0])
        focal_length_y = float(cam_params['intrinsics_wo_distortion']['f'][1])
        camera_center_x = float(cam_params['intrinsics_wo_distortion']['c'][0])
        camera_center_y = float(cam_params['intrinsics_wo_distortion']['c'][1])
        
        with open(sh_file, 'a+') as f:
            f.write('python prox/main.py --config cfg_files/RGB.yaml --recording_dir {0} --output_folder ./output --vposer_ckpt ./models/vposer_v1_0/ --part_segm_fn ./models/smplx_parts_segm.pkl --model_folder ./models --focal_length_x {1} --focal_length_y {2} --camera_center_x {3} --camera_center_y {4}'.format(recording_dir_path, focal_length_x, focal_length_y, camera_center_x, camera_center_y) + '\n')

if __name__ == '__main__':
    input_path = './input'
    sh_file = './prox.sh'
    generate_sh(input_path=input_path, sh_file=sh_file)
