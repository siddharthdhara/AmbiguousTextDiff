import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import pandas as pd
from tqdm import tqdm

sys.path.append(".")  # Add current directory to path

from guided_diffusion import dist_util, logger
from guided_diffusion.text_encoder import TextEncoder
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def main():
    args = create_argparser().parse_args()

    # Set up distributed processes and logging
    dist_util.setup_dist()
    logger.configure(dir=args.output_dir)

    # Load text encoder
    logger.log("Creating text encoder...")
    text_encoder = TextEncoder(
        model_name=args.text_model_name,
        max_length=args.max_text_length,
        output_dim=args.text_dim
    )
    text_encoder.to(dist_util.dev())
    text_encoder.eval()

    # Load model and diffusion
    logger.log("Creating model and diffusion...")
    model_diffusion_args = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion, _, _ = create_model_and_diffusion(**model_diffusion_args)

    # 1. Move the fresh model to the correct device
    model.to(dist_util.dev())
    
    # 2. Perform a dummy forward pass to initialize the dynamic layers
    logger.log("Initializing dynamic model layers...")
    with torch.no_grad():
        dummy_x = torch.zeros(
            1, 
            args.image_actual_channels + 1, 
            args.image_size, 
            args.image_size, 
            device=dist_util.dev()
        )
        dummy_ts = torch.zeros(1, device=dist_util.dev()).long()
        dummy_text_emb = torch.zeros(1, args.text_dim, device=dist_util.dev())
        model(dummy_x, dummy_ts, text_embeddings=dummy_text_emb)
        
    # 3. NOW load the state dict, since the model architecture is complete
    logger.log(f"Loading model weights from {args.model_path}...")
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    
    # 4. Set the model to evaluation mode
    model.eval()

    # Load test images and text
    logger.log("Loading test data...")

    # Load image paths
    if args.test_list:
        with open(args.test_list, 'r') as f:
            image_paths = [line.strip() for line in f.readlines()]
    else:
        image_paths = [args.input_image]

    # Load text annotations
    if args.text_annotations_file:
        annotations_df = pd.read_csv(args.text_annotations_file)
        text_dict = {}
        for _, row in annotations_df.iterrows():
            img_id = row['file_name']
            text = row['Text']
            text_dict[img_id] = text

    os.makedirs(args.output_dir, exist_ok=True)

    # Generate multiple segmentation samples for each image
    logger.log(f"Generating {args.num_samples} segmentation samples for each image...")
    for image_path in tqdm(image_paths):
        path_parts = os.path.normpath(image_path).split(os.sep)
        if len(path_parts) >= 4:  # Need at least patient, nodule, images, slice.png
            slice_file_basename = path_parts[-1]
            nodule_dir_name = path_parts[-3]
            patient_dir_name = path_parts[-4]  # This should be like "LIDC-IDRI-0001"

            img_id_for_saving = f"{patient_dir_name}_{nodule_dir_name}_{slice_file_basename.split('.')[0]}"
            patient_id_for_text_lookup = patient_dir_name
        else:  # Fallback if path is not as expected (e.g., just a filename)
            base_name_for_saving = os.path.basename(image_path).split('.')[0]
            img_id_for_saving = base_name_for_saving
            # Try to extract patient ID from base_name_for_saving if it follows "LIDC-IDRI-XXXX_..."
            if base_name_for_saving.startswith("LIDC-IDRI-") and "_" in base_name_for_saving:
                patient_id_for_text_lookup = base_name_for_saving.split("_")[0]  # Crude extraction
            else:  # Or if it's just LIDC-IDRI-XXXX
                patient_id_for_text_lookup = base_name_for_saving  # Assume it *is* the patient ID
            logger.log(f"Warning: Could not parse full patient/nodule/slice from path '{image_path}'. Using '{img_id_for_saving}' for saving and '{patient_id_for_text_lookup}' for text lookup.")

        # Load image
        # image = Image.open(image_path).convert("RGB")
        image = Image.open(image_path).convert("L") # Load as grayscale (1 channel)
        image = image.resize((args.image_size, args.image_size))
        
        # For a grayscale image, the process is slightly different (no permute needed)
        image_np = np.array(image) / 127.5 - 1.0
        # Add channel dimension before batch dimension
        image_tensor = torch.from_numpy(image_np).unsqueeze(0).float() 
        image_tensor = image_tensor.unsqueeze(0).to(dist_util.dev()) # [1, 1, H, W]

        # Get text description
        if args.text_annotations_file:
            text = text_dict.get(patient_id_for_text_lookup, args.default_text)
        else:
            text = args.default_text

        # Generate text embeddings
        #text_embedding = text_encoder([text])
        #text_embedding = text_embedding.to(dist_util.dev())
        
        if args.disable_text_guidance:
            model_diffusion_args["use_text_guidance"] = False
        # Empty text embedding (zeros) with proper shape
            text_embedding = torch.zeros((1, args.text_dim), device=dist_util.dev())
        else:
        # Generate text embeddings
            text_embedding = text_encoder([text])
            text_embedding = text_embedding.to(dist_util.dev())

        # model_kwargs for p_sample_loop
        model_kwargs_for_sampling = {"text_embeddings": text_embedding}
        if args.class_cond:  # If you add class conditioning during sampling
            # model_kwargs_for_sampling["y"] = ... some class label tensor
            pass

        # Generate multiple samples
        all_samples_mask_only = []  # Store only the mask part
        for sample_idx in range(args.num_samples):
            # Create initial noise for the MASK channel ONLY
            # Determine number of image channels the model expects its input image part to have
            # This should align with image_actual_channels used in training
            # This example assumes model trained with RGB images
            num_model_image_channels = 1

            mask_noise_shape = (1, 1, args.image_size, args.image_size)  # Assuming 1 channel mask
            mask_start_noise = torch.randn(mask_noise_shape, device=dist_util.dev())

            # Ensure image_tensor (the clean input image) has num_model_image_channels
            # image_tensor from PIL (RGB) is [1, 3, H, W]
            # If num_model_image_channels is different, adjust image_tensor here.
            # For now, assume image_tensor is correctly [1, num_model_image_channels, H, W]

            # Concatenate the clean image with the mask noise for t=T
            initial_x_T_for_loop = torch.cat((image_tensor, mask_start_noise), dim=1)
            initial_x_T_for_loop = initial_x_T_for_loop.to(dist_util.dev())

            # Sample from the model
            # The shape passed to p_sample_loop should be the shape of initial_x_T_for_loop
            # The 'noise' argument to p_sample_loop is this initial_x_T_for_loop
            generated_mask_sample = diffusion.p_sample_loop(
                model,
                initial_x_T_for_loop.shape,  # Shape of (image + mask_noise)
                noise=initial_x_T_for_loop,  # Starting state for the reverse process
                clip_denoised=args.clip_denoised,  # Clipping applied to mask's x0_pred
                # denoised_fn: if you have one for masks
                model_kwargs=model_kwargs_for_sampling,
                # device=dist_util.dev(), # p_sample_loop_progressive handles device
                # progress=True, # Optional for debugging a single sample generation
            )
            # generated_mask_sample is the final x0 for the MASK channel

            # Post-process sample - threshold or sigmoid (applied to the mask)
            if args.threshold > 0:
                processed_mask_sample = (generated_mask_sample > args.threshold).float()
            else:  # Assuming output is logits-like for mask, apply sigmoid
                processed_mask_sample = torch.sigmoid(generated_mask_sample)

            sample_np_mask_only = processed_mask_sample.cpu().numpy()
            all_samples_mask_only.append(sample_np_mask_only)

            # Save individual sample (mask only)
            sample_img_mask_only = (sample_np_mask_only[0, 0] * 255).astype(np.uint8)  # Assuming [1,1,H,W]
            sample_pil_mask_only = Image.fromarray(sample_img_mask_only)
            sample_path = os.path.join(args.output_dir, f"{img_id_for_saving}_sample_{sample_idx}.png")
            sample_pil_mask_only.save(sample_path)

        # Ensemble predictions (using all_samples_mask_only)
        if args.save_ensemble:
            # Stack masks for averaging
            stacked_masks = np.stack([s[0] for s in all_samples_mask_only], axis=0)  # [N, 1, H, W]

            # Average prediction
            avg_pred_mask_channel = np.mean(stacked_masks, axis=0)  # [1, H, W]
            avg_img = (avg_pred_mask_channel[0] * 255).astype(np.uint8)  # Take the single channel
            avg_pil = Image.fromarray(avg_img)
            avg_path = os.path.join(args.output_dir, f"{img_id_for_saving}_ensemble_avg.png")
            avg_pil.save(avg_path)

            # Majority voting
            if args.threshold > 0:
                maj_pred_mask_channel = (np.mean(stacked_masks, axis=0) > 0.5).astype(np.float32)  # [1, H, W]
                maj_img = (maj_pred_mask_channel[0] * 255).astype(np.uint8)
                maj_pil = Image.fromarray(maj_img)
                maj_path = os.path.join(args.output_dir, f"{img_id_for_saving}_ensemble_majority.png")
                maj_pil.save(maj_path)

        # Save input image for reference
        input_img = Image.fromarray((np.array(image)))
        input_path = os.path.join(args.output_dir, f"{img_id_for_saving}_input.png")
        input_img.save(input_path)

        # Save text annotation
        with open(os.path.join(args.output_dir, f"{img_id_for_saving}_text.txt"), "w") as f:
            f.write(text)

    logger.log("Sampling complete.")

def create_argparser():
    defaults = dict(
        model_path="",
        input_image="",
        test_list="",
        output_dir="",
        text_annotations_file="",
        default_text="A lung nodule in CT scan.",
        num_samples=5,
        clip_denoised=True,
        threshold=0.5,
        save_ensemble=True,
        text_model_name="emilyalsentzer/Bio_ClinicalBERT",
        max_text_length=128,
        text_dim=768,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable_text_guidance", action="store_true", help="Disable text guidance for ablation")
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
