#!/usr/bin/env python3
"""
Batch inference script for DICEPTION model.
Processes multiple images from a JSON file containing task specifications.
"""

import os
import torch
import argparse
import matplotlib

from PIL import Image
from datasets import load_dataset
from models.Renderer import RenderNet
from utils.files_op import ensure_directory
from dataset.dataset import DataCollator_MT_EVAL
from utils.dist import move_to_cuda, fp32_to_bf16

import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F


def chw2hwc(chw):
    if isinstance(chw, torch.Tensor):
        hwc = torch.permute(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    return hwc

def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.cpu().float().detach().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    depth = depth.mean(0)
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0
        
    img_colored = img_colored_np

    return img_colored


def save(inputs, outputs, args, save_idx):
    input_imgs = []
    pred_imgs = []
    has_target = False

    if "input_to_viz" in inputs.keys():
        input_imgs.append(inputs["input_to_viz"] * 0.5 + 0.5)
    else:
        input_imgs.append(inputs["input_images"] * 0.5 + 0.5)
    pred_imgs.append(outputs["images"])
    types = inputs['prompt']
    paths = inputs['path']

    input_imgs = torch.cat(input_imgs, dim=0)
    pred_imgs = torch.cat(pred_imgs, dim=0)
    
    
    toimg = transforms.ToPILImage()
    if 'target_images' in inputs.keys():
        # has_target = True
        gt_imgs = [inputs["target_images"] * 0.5 + 0.5]
        gt_imgs = torch.cat(gt_imgs, dim=0)
        gt_imgs = [toimg(img.to(torch.float16).cpu()) for img in gt_imgs]
    
    images = [toimg(img.to(torch.float16).cpu()) for img in pred_imgs]
    
    if 'depth' in types[0]:
        np_images = [img.mean(dim=0) for img in pred_imgs]
    else:
        np_images = [img for img in pred_imgs]
    inp_images = [toimg(img.to(torch.float16).cpu()) for img in input_imgs]
    original_size = inputs['original_size']

    image_path = os.path.join(args.save_path, 'images')
    ensure_directory(os.path.join(image_path))
    
    for i in range(len(images)):

        ori_w, ori_h = original_size[i]

        if has_target:
            result = Image.new('RGB', (ori_w * 3, ori_h))
        else:
            result = Image.new('RGB', (ori_w * 2, ori_h))

        if 'depth' in types[i]:
            depth_colored = colorize_depth_maps(
                pred_imgs[i], 0, 1
            ).squeeze()  # [3, H, W], value in (0, 1)
            depth_colored = (depth_colored * 255).astype(np.uint8)
            depth_colored_hwc = chw2hwc(depth_colored)
            images[i] = Image.fromarray(depth_colored_hwc)
            
            if has_target:
                depth_colored = colorize_depth_maps(
                    gt_imgs[i], 0, 1
                ).squeeze()  # [3, H, W], value in (0, 1)
                depth_colored = (depth_colored * 255).astype(np.uint8)
                depth_colored_hwc = chw2hwc(depth_colored)
                gt_imgs[i] = Image.fromarray(depth_colored_hwc)


        result.paste(inp_images[i].resize((ori_w, ori_h)), (0, 0))
        result.paste(images[i].resize((ori_w, ori_h)), (ori_w, 0))
        if has_target:
            result.paste(gt_imgs[i].resize((ori_w, ori_h)), (2 * ori_w, 0))

        cur_id = paths[i].split('./')[-1].split('.')[0].split('/')[-1]

        save_path = os.path.join(image_path, f'{cur_id}_{types[i]}.png')
        # save_path = os.path.join(image_path, f'{cur_id}.png')
        save_path_np = os.path.join(image_path, f'{cur_id}.npy')
        
        parent_dir = os.path.dirname(save_path)
        ensure_directory(parent_dir)

        if args.save_npy:
            
            if 'depth' in types[i]:
                resized_depth = F.interpolate(np_images[i].unsqueeze(0).unsqueeze(0), size=(ori_h, ori_w), mode='bilinear', align_corners=False).squeeze()
                np.save(save_path_np, resized_depth.to(torch.float16).cpu().detach().numpy())
            elif 'normal' in types[i]:
                normal = F.interpolate(np_images[i].unsqueeze(0), size=(ori_h, ori_w), mode='bilinear', align_corners=False).squeeze()
                np.save(save_path_np, normal.to(torch.float16).cpu().detach().numpy())

        result.save(save_path)
    save_idx += len(images)
    return save_idx

def batch_inference(model, args):
    """
    Perform batch inference on a dataset specified in JSON format.
    
    Args:
        model: The loaded RenderNet model
        args: Command line arguments containing paths and parameters
    """
    generator = torch.Generator(device=model.device)
    generator.manual_seed(args.seed)

    save_idx = 0

    print(f"Loading dataset from: {args.input_path}")
    torch.set_float32_matmul_precision('high')
    
    # Setup data transforms
    norm = transforms.Normalize([0.5], [0.5])
    resize = transforms.Resize([768, 768], interpolation=transforms.InterpolationMode.BILINEAR)

    # Load dataset from JSON
    dataset = load_dataset('json', data_files=args.input_path, cache_dir="./cache")
    dataset = dataset['train']
    
    # Create data collator
    collate_fn_eval = DataCollator_MT_EVAL(
        resize, norm, 768, 768, 
        args.data_root_path, 
        args.go_through_all_seg_labels
    )
    
    # Handle segmentation labels processing
    if args.go_through_all_seg_labels:
        args.new_batch_size = args.batch_size
        args.batch_size = 1
        
    # Create dataloader
    eval_dataloader = torch.utils.data.DataLoader(
        dataset, 
        num_workers=0, 
        batch_size=args.batch_size, 
        collate_fn=collate_fn_eval
    )

    print(f"Processing {len(dataset)} samples with batch size {args.batch_size}")
    
    # Process batches
    for batch_idx, inputs in enumerate(eval_dataloader):
        print(f"Processing batch {batch_idx + 1}/{len(eval_dataloader)}")
        
        if args.go_through_all_seg_labels:
            total_batch = inputs['input_to_viz'].shape[0]
            
            # Process in chunks for memory efficiency
            for i in range(0, total_batch, args.new_batch_size):
                outputs = {}
                chunk = {key: value[i:i + args.new_batch_size] for key, value in inputs.items()}
                chunk['generator'] = generator
                chunk = move_to_cuda(chunk)
                chunk = fp32_to_bf16(chunk)
                outputs = model.eval_fn(chunk, outputs)
                save_idx = save(chunk, outputs, args, save_idx)
        
        else:
            outputs = {}
            inputs['generator'] = generator

            inputs = move_to_cuda(inputs)
            inputs = fp32_to_bf16(inputs)

            outputs = model.eval_fn(inputs, outputs)
            save_idx = save(inputs, outputs, args, save_idx)

    print(f"Batch inference completed! Processed {save_idx} images.")
    print(f"Results saved to: {args.save_path}")


def main():
    parser = argparse.ArgumentParser(description="DICEPTION Batch Inference")
    
    # Model and data paths
    parser.add_argument('--pretrained_model_path', required=True, type=str, 
                       help='Path to pretrained SD3 model')
    parser.add_argument('--diception_path', required=True, type=str,
                       help='Path to DICEPTION_v1.pth weights')
    parser.add_argument('--input_path', required=True, type=str,
                       help='Path to JSON file containing dataset for batch inference')
    parser.add_argument('--data_root_path', default='/', type=str,
                       help='Root path for data files referenced in JSON')
    parser.add_argument('--save_path', default='./batch_results', type=str,
                       help='Directory to save inference results')
    
    # Inference parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--guidance_scale', type=float, default=2.0, help='Guidance scale')
    parser.add_argument('--num_inference_steps', type=int, default=28, help='Number of inference steps')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    
    # Output options
    parser.add_argument("--save_npy", action="store_true", 
                       help="Save numpy arrays for depth/normal outputs")
    parser.add_argument("--go_through_all_seg_labels", action="store_true",
                       help="Process all segmentation labels (for interactive segmentation)")
    
    # Legacy compatibility parameters (kept for backward compatibility)
    parser.add_argument('--logit_mean', type=float, default=0.0)
    parser.add_argument('--logit_std', type=float, default=1.0)
    parser.add_argument('--mode_scale', type=float, default=1.29)
    parser.add_argument("--weighting_scheme", default="logit_normal")

    args = parser.parse_args()

    print("=" * 60)
    print("DICEPTION Batch Inference")
    print("=" * 60)
    print(f"Model path: {args.pretrained_model_path}")
    print(f"Weights path: {args.diception_path}")
    print(f"Input JSON: {args.input_path}")
    print(f"Output directory: {args.save_path}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)

    # Load model
    print("Loading model...")
    model = RenderNet(args)
    model = model.to('cuda', torch.bfloat16)
    
    # Load trained weights
    print("Loading trained weights...")
    ckpt = torch.load(args.diception_path, map_location='cuda')
    warnings = model.transformer.load_state_dict(ckpt['transformer'], strict=False)
    if warnings.missing_keys or warnings.unexpected_keys:
        print(f"Transformer loading warnings: {warnings}")
    warnings = model.point_embedder.load_state_dict(ckpt['point_embedder'], strict=False)
    if warnings.missing_keys or warnings.unexpected_keys:
        print(f"Point embedder loading warnings: {warnings}")
    warnings = model.not_seg_embeddings.load_state_dict(ckpt['not_seg_embeddings'], strict=False)
    if warnings.missing_keys or warnings.unexpected_keys:
        print(f"Not seg embeddings loading warnings: {warnings}")
    warnings = model.seg_embeddings.load_state_dict(ckpt['seg_embeddings'], strict=False)
    if warnings.missing_keys or warnings.unexpected_keys:
        print(f"Seg embeddings loading warnings: {warnings}")
    model.positional_encoding_gaussian_matrix = ckpt['positional_encoding_gaussian_matrix']
    
    # Set model to inference mode
    model.set_inference_mode()
    print("Model loaded successfully!")

    # Run batch inference
    with torch.no_grad():
        batch_inference(model, args)


if __name__ == '__main__':
    main()
        
