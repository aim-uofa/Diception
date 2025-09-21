#!/usr/bin/env python3
"""
Simple inference script for DICEPTION model.
This script provides an easy-to-use interface for running inference on images.
"""

import os
import torch
import argparse
from PIL import Image
from models.Renderer import RenderNet
from utils.files_op import ensure_directory
from utils.dist import move_to_cuda, fp32_to_bf16
import torchvision.transforms as transforms
import numpy as np
from batch_inference import colorize_depth_maps, chw2hwc


def run_inference(model, image_path, prompt, output_dir="./outputs", seed=42, coor_points=None):
    """
    Run inference on a single image.
    
    Args:
        model: The loaded RenderNet model
        image_path: Path to input image
        prompt: Task prompt (e.g., '[[image2depth]]', etc.)
        output_dir: Directory to save results
        seed: Random seed for reproducibility
        coor_points: List of coordinate points in format [(y1,x1), (y2,x2), ...] 
                    where coordinates are normalized to [0,1] range
    """
    # Load and preprocess image
    orig_image = Image.open(image_path).convert('RGB')
    w, h = orig_image.size
    
    # Resize to model input size
    image = orig_image.resize((768, 768), Image.LANCZOS)
    to_tensor = transforms.ToTensor()
    image = (to_tensor(image) - 0.5) * 2
    
    # Prepare point coordinates
    coor_point = torch.zeros((1, 5, 2))
    point_labels = torch.zeros((1, 5, 1))
    
    # Fill in provided coordinates if any
    if coor_points is not None:
        num_points = min(len(coor_points), 5)  # Max 5 points supported
        for i, (y, x) in enumerate(coor_points[:num_points]):
            # Convert (y,x) format to (x,y) format for the model
            coor_point[0, i, 0] = float(x)  # x coordinate
            coor_point[0, i, 1] = float(y)  # y coordinate
            point_labels[0, i, 0] = 1.0     # Set label to 1 for provided points
    
    # Create input dictionary
    inputs = {
        'input_images': image.unsqueeze(0),
        'original_size': torch.tensor([[w, h]]),
        'target_size': torch.tensor([[768, 768]]),
        'prompt': [prompt],
        'coor_point': coor_point,
        'point_labels': point_labels,
        'path': os.path.basename(image_path)
    }
    
    # Move to device and convert dtype
    inputs = move_to_cuda(inputs)
    inputs = fp32_to_bf16(inputs)
    
    # Run inference
    with torch.no_grad():
        outputs = model(inputs)
    
    # Save results
    ensure_directory(output_dir)
    result_tensor = outputs['images'][0]

    # Prepare output visualization
    if 'depth' in prompt:
        depth_colored = colorize_depth_maps(
            result_tensor, 0, 1
        ).squeeze()  # [3, H, W], value in (0, 1)
        depth_colored = (depth_colored * 255).astype(np.uint8)
        depth_colored_hwc = chw2hwc(depth_colored)
        result_pil = Image.fromarray(depth_colored_hwc)
    else:
        result_vis = (result_tensor.cpu().clamp(0, 1) * 255).byte()
        result_pil = transforms.ToPILImage()(result_vis)

    # Resize to original and concatenate: input left, output right
    result_pil = result_pil.resize((w, h), Image.LANCZOS)
    concat = Image.new('RGB', (w * 2, h))
    concat.paste(orig_image.resize((w, h)), (0, 0))
    concat.paste(result_pil, (w, 0))

    output_path = os.path.join(output_dir, f"result_{os.path.basename(image_path)}")
    concat.save(output_path)
    print(f"Result saved to: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="DICEPTION Inference")
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--prompt', required=True, 
                       help='Task prompt (e.g., [[image2depth]], [[image2normal]])')
    parser.add_argument('--pretrained_model_path', required=True, help='Path to pretrained model')
    parser.add_argument('--diception_path', required=True, help='Path to DICEPTION_v1.pth weights')
    parser.add_argument('--output_dir', default='./outputs', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--guidance_scale', type=float, default=1.0, help='Guidance scale')
    parser.add_argument('--num_inference_steps', type=int, default=1, help='Number of inference steps')
    parser.add_argument('--points', type=str, help='Coordinate points in format "y1,x1;y2,x2;y3,x3" (normalized to [0,1], used in interactive segmentation)')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model = RenderNet(args)
    model = model.to('cuda', torch.bfloat16)
    
    # Load trained weights
    print("Loading trained weights...")
    ckpt = torch.load(args.diception_path, map_location='cuda')
    model.transformer.load_state_dict(ckpt['transformer'], strict=False)
    model.point_embedder.load_state_dict(ckpt['point_embedder'], strict=False)
    model.not_seg_embeddings.load_state_dict(ckpt['not_seg_embeddings'], strict=False)
    model.seg_embeddings.load_state_dict(ckpt['seg_embeddings'], strict=False)
    model.positional_encoding_gaussian_matrix = ckpt['positional_encoding_gaussian_matrix']
    
    # Set to inference mode
    model.set_inference_mode()
    print("Model loaded successfully!")
    
    # Run inference
    print(f"Running inference on {args.image} with prompt '{args.prompt}'...")
    # Parse coordinate points if provided
    coor_points = None
    if args.points:
        try:
            # Parse points in format "y1,x1;y2,x2;y3,x3"
            point_strs = args.points.split(';')
            coor_points = []
            for point_str in point_strs:
                y, x = map(float, point_str.split(','))
                coor_points.append((y, x))
            print(f"Using coordinate points: {coor_points}")
        except Exception as e:
            print(f"Error parsing points: {e}")
            print("Points should be in format 'y1,x1;y2,x2;y3,x3' with coordinates normalized to [0,1]")
            return
    
    output_path = run_inference(
        model, 
        args.image, 
        args.prompt, 
        args.output_dir, 
        args.seed,
        coor_points
    )
    
    print("Inference completed!")


if __name__ == "__main__":
    main()
