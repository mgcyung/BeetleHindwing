from share import *
import config
import json
import shutil

import cv2
from PIL import Image
import einops
# import gradio as gr
import numpy as np
import torch
import random
import os
import glob
from skimage.transform import PiecewiseAffineTransform, warp

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from annotator.midas import MidasDetector
from annotator.hed import HEDdetector
from annotator.mlsd import MLSDdetector
from annotator.lineart import LineartDetector
from annotator.normalbae import NormalBaeDetector

from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


apply_canny = CannyDetector()
apply_midas = MidasDetector()
apply_hed = HEDdetector()
apply_mlsd = MLSDdetector()
apply_lineart = LineartDetector()
apply_normalbae = NormalBaeDetector()

annotator_dict ={'canny': apply_canny,
                 'depth': apply_midas,
                 'HED': apply_hed,
                 'normal': apply_normalbae,
                 # 'hough': apply_mlsd,
                 'lineart': apply_lineart,
                 'scribble': apply_hed}

weight_dict = {'canny': './models/control_v11p_sd15_canny.pth',
               'depth': './models/control_v11f1p_sd15_depth.pth',
               'HED': './models/control_v11p_sd15_softedge.pth',
               'normal': './models/control_v11p_sd15_normalbae.pth',
               # 'hough': './models/control_v11p_sd15_mlsd.pth',
               'lineart': './models/control_v11p_sd15_lineart.pth',
               'scribble': './models/control_v11p_sd15_scribble.pth'}

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/v1-5-pruned.ckpt', location='cuda'), strict=False)
# model = model.cuda()
# ddim_sampler = DDIMSampler(model)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold, tps, annotator='canny'):
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        apply_annotator = annotator_dict[annotator]
        if annotator == 'canny':
            detected_map = apply_annotator(img, low_threshold, high_threshold)
        elif annotator == 'hough':
            detected_map = apply_annotator(img, 0.1, 0.1)
        elif annotator == 'lineart':
            detected_map = apply_annotator(img, coarse=False)
        else:
            detected_map = apply_annotator(img)
        detected_map = HWC3(detected_map)
        # Thin Plate Spline transform
        detected_map = warp(detected_map, tps)
        detected_map = detected_map * 255.0

        if annotator == 'lineart':
            control = 1.0 - torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        else:
            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        global ddim_sampler
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [(detected_map).astype(np.uint8)] + results


def batch_process(src_dir, dst_dir):
    for src_img_path in glob.glob(os.path.join(src_dir, '*.png')):
        # resize each image
        species_name = os.path.basename(src_img_path)[:-4]
        dst_img_path = os.path.join(dst_dir, species_name + '.png')
        print(src_img_path)
        img_src = Image.open(src_img_path)
        img_src = np.array(img_src)
        results = one_image_process(img_src)
        img_dst = Image.fromarray(results[1])
        canny_tps = Image.fromarray(results[0])
        img_dst.save(dst_img_path)
        canny_tps.save(dst_img_path[:-4] + '_canny-tps.png')


def json_process(annotations_file, src_dir, dst_dir, num_tps=1, sigma=10):
    src_annotations_folder = f'{src_dir}/annotations'
    annotations_file_path = '{}/{}'.format(src_annotations_folder, annotations_file)
    # 读取 coco 格式的数据集：
    with open(annotations_file_path, 'r') as f:
        data = json.load(f)

    annotations = data['annotations']
    src_image_folder = f'{src_dir}/images'
    # 新建数据集图像目录
    dst_image_folder = f'{dst_dir}/images'
    shutil.rmtree(dst_image_folder, ignore_errors=True)
    os.makedirs(dst_image_folder, exist_ok=False)
    # 生成采样后的数据
    dst_data = {
        'images': [],
        'annotations': []
    }

    global model
    global ddim_sampler
    for annotator in annotator_dict.keys():

        model.load_state_dict(load_state_dict(weight_dict[annotator], location='cuda'), strict=False)
        model = model.cuda()
        ddim_sampler = DDIMSampler(model)

        dst_annotator_image_folder = f'{dst_image_folder}/{annotator}'
        shutil.rmtree(dst_annotator_image_folder, ignore_errors=True)

        for i, annotation in enumerate(data['annotations']):

            # Load image
            src_image_id = annotation['image_id']
            src_image = [image_info.copy() for image_info in data['images'] if image_info['id']==src_image_id][0].copy()
            src_image_file_name = src_image['file_name']
            src_img_path = '{}/{}'.format(src_image_folder, src_image_file_name)
            # Image tps & Generate
            print(src_img_path)
            img_src = Image.open(src_img_path)
            width = img_src.width
            height = img_src.height
            img_src = np.array(img_src)

            for j in range(num_tps):

                dst_annotation = annotation.copy()
                dst_image = src_image.copy()
                dst_image_id = i * num_tps + j
                dst_image_file_name = '{}_{:03d}.png'.format(src_image_file_name[:-4], j)
                dst_img_path = '{}/{}'.format(dst_annotator_image_folder, dst_image_file_name)
                dst_image_subdir = os.path.dirname(dst_image_file_name)
                os.makedirs(os.path.join(dst_annotator_image_folder, dst_image_subdir), exist_ok=True)

                # Landmark offset & generate tps
                src_keypoints = dst_annotation['keypoints']
                src_lankmarks = np.array([[src_keypoints[k*3+0], src_keypoints[k*3+1]] for k in range(36)])
                offset = np.random.randint(-sigma, sigma, [36, 2])
                dst_landmarks = src_lankmarks + offset
                dst_keypoints = src_keypoints.copy()
                for l in range(36):
                    dst_keypoints[l*3] = dst_landmarks[l][0]
                    dst_keypoints[l*3 + 1] = dst_landmarks[l][1]
                tps = generate_tps(src_lankmarks, offset, height, width)

                # Generate image
                results = one_image_process(img_src, tps, annotator=annotator)
                img_dst = Image.fromarray(results[1])
                canny_tps = Image.fromarray(results[0])
                img_dst.save(dst_img_path)
                canny_tps.save(dst_img_path[:-4] + '_' + annotator + '-tps.png'.format(j))

                # save to dst dir
                dst_image['file_name'] = dst_image_file_name
                dst_data['images'].append(dst_image)
                dst_annotation['image_id'] = dst_image_id
                dst_annotation['keypoints'] = dst_keypoints
                dst_annotation['bbox'] = [0, 0, width, height]
                dst_annotation['center'] =[width/2, height/2]
                dst_data['annotations'].append(dst_annotation)

    # 生成新的 coco 格式的数据集：
    dst_annotations_folder = f'{dst_dir}/annotations'
    shutil.rmtree(dst_annotations_folder, ignore_errors=True)
    os.makedirs(dst_annotations_folder, exist_ok=False)
    with open(f'{dst_annotations_folder}/train.json', 'w') as f:
        json.dump(dst_data, f)


def one_image_process(img_src, tps, annotator='canny'):

    input_image = img_src
    prompt = ''
    a_prompt = 'A hindwing extracted from body'
    n_prompt = ''
    num_samples = 1
    image_resolution = 512
    ddim_steps = 20
    guess_mode = False
    strength = 1.0
    scale = 9.0
    seed = random.randint(0, 2147483647)
    eta = 0.0
    low_threshold = 100
    high_threshold = 200

    # tps = generate_tps(512, 1024)
    results = process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold, tps, annotator=annotator)
    print(len(results))
    print(results[0].shape)
    # for i, result in enumerate(results):
        # img = Image.fromarray(result)
        # img.save(f'result_{i}.png')
    return results


def generate_tps(landmarks, offsets, height, width):

    # 原始图像中的5个点的坐标
    src_pts = np.array(landmarks, dtype=np.float32)
    # 对应的新图像中的坐标
    dst_pts = np.array([coord + offset for coord, offset in zip(landmarks, offsets)], dtype=np.float32)
    # 扩展新坐标系的边界点
    src_pts_ext = np.vstack((src_pts, [[0, 0], [0, height], [width, 0], [width, height]]))
    dst_pts_ext = np.vstack((dst_pts, [[0, 0], [0, height], [width, 0], [width, height]]))

    # 计算 TPS 变换
    tps = PiecewiseAffineTransform()
    tps.estimate(src_pts_ext, dst_pts_ext)

    return tps


if __name__ == '__main__':

    # src_dir = '../datasets/FCHWL_SIZES-RP-0_PNG'
    # dst_dir = '../outputs/FCHWL_SIZES-RP-0_PNG_tps'
    # os.makedirs(dst_dir, exist_ok=True)
    # batch_process(src_dir, dst_dir)

    width = 1024
    height = 512

    annotations_file = 'train.json'
    src_dir = '../../data/LBHW3_resized'
    # dst_dir = '../../data/LBHW3_resized_tps'
    dst_dir = '../../data/LBHW3_resized_v1.1_tps10'
    json_process(annotations_file, src_dir, dst_dir, num_tps=10)
