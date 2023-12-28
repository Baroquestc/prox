
import pyrender
import numpy as np
import trimesh
import cv2
import PIL.Image as pil_img
from cmd_parser import parse_config
from icecream import ic
args = parse_config()

if __name__ == '__main__':
    camera_center = np.array([args['camera_center_x'], args['camera_center_y']])
    camera_pose = np.eye(4)
    camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
    camera = pyrender.camera.IntrinsicsCamera(
        fx=args['focal_length_x'], fy=args['focal_length_y'],
        cx=camera_center[0], cy=camera_center[1])
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(219/255., 112/255., 147/255, 1.0))

    out_mesh_fn = './RGB/BBQ_001_guitar_01_00135/meshes/000.ply'
    out_mesh = trimesh.load(out_mesh_fn)  
    body_mesh = pyrender.Mesh.from_trimesh(
        out_mesh, material=material)
    
    ## rendering body
    img = cv2.imread('../data/prox/recordings/BBQ_001_guitar_01_00135/Color/BBQ_001_guitar_01_00135.png')
    # 图片左右翻转
    # img = img[:, ::-1, :]
    H, W, _ = img.shape

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                            ambient_light=(0.3, 0.3, 0.3))
    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)
    # for node in light_nodes:
    #     scene.add_node(node)

    scene.add(body_mesh, 'mesh')

    r = pyrender.OffscreenRenderer(viewport_width=W,
                                    viewport_height=H,
                                    point_size=1.0)
    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0

    ic(color.shape)
    valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
    # valid_mask = (color[:, :, ] > 0)[:, :, np.newaxis]
    # valid_mask = (color[:, :, ] > 0)[..., np.newaxis]
    ic(valid_mask.shape)

    # img = pil_img.fromarray((img * 255).astype(np.uint8))
    # img.save(out_img_fn)
    # exit()

    input_img = img
    ic(input_img.shape, valid_mask.shape, color[:, :, :-1].shape)
    ic((color[:, :, :] * valid_mask).shape)
    ic(((1- valid_mask) * input_img).shape)

    output_img = (color[:, :, :] * valid_mask +
                    (1-valid_mask) * input_img)

    # output_img = (color[:, :, :-1] * valid_mask +
    #               (1-valid_mask) * input_img)

    img = pil_img.fromarray((output_img * 255).astype(np.uint8))
    img.save('../BBQ_001_guitar_01_00135.png')