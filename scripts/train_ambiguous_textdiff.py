import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from PIL import Image
import glob


# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(".")  # Add current directory to path

from guided_diffusion import dist_util, logger
from guided_diffusion.text_encoder import TextEncoder
from guided_diffusion.unet import TextGuidedUNetModel
from guided_diffusion.dataset import TextGuidedImageSegDataset
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

def main():
    args = create_argparser().parse_args()

    # Set up distributed training if needed
    dist_util.setup_dist()
    logger.configure(dir=args.log_dir)

    # Load dataset (adjust paths as needed)
    # image_paths = glob.glob(os.path.join(args.data_dir, "images/*.png"))
    # mask_paths = [p.replace("images", "masks") for p in image_paths]

    logger.log("Loading dataset...")

    # Load text annotations from CSV
    annotations_df = pd.read_csv(args.text_annotations_file)

    # Map image IDs to text annotations
    text_dict = {}
    for _, row in annotations_df.iterrows():
        img_id_csv = row['file_name'] # e.g., LIDC-IDRI-0001_nodule-0_slice-0
        text = row['Text']
        text_dict[img_id_csv] = text

    # Prepare lists for dataset
    all_image_paths = []
    all_mask_paths_lists = [] # This will be a list of lists
    all_text_annotations = []

    base_data_dir = args.data_dir # e.g., "path/to/LIDC-IDRI-slices"
    logger.log(f"Scanning data directory: {base_data_dir}")

    patient_dirs = sorted(glob.glob(os.path.join(base_data_dir, "LIDC-IDRI-*")))
    if not patient_dirs:
        logger.log(f"ERROR: No patient directories found in {base_data_dir} matching LIDC-IDRI-*")
        sys.exit(1)

    for patient_dir_path in patient_dirs:
        patient_id = os.path.basename(patient_dir_path)
        nodule_dirs = sorted(glob.glob(os.path.join(patient_dir_path, "nodule-*")))
        for nodule_dir_path in nodule_dirs:
            nodule_id_only = os.path.basename(nodule_dir_path) # e.g., "nodule-0"

            images_sub_dir = os.path.join(nodule_dir_path, "images")
            slice_image_files = sorted(glob.glob(os.path.join(images_sub_dir, "slice-*.png")))

            for slice_image_file_path in slice_image_files:
                slice_basename = os.path.basename(slice_image_file_path) # e.g., "slice-0.png"
                slice_id_only = slice_basename.split('.')[0] # e.g., "slice-0"

                # Construct the unique image_id for text lookup and internal use
                # This MUST match the format in your CSV's 'image_id' column
                current_image_id = f"{patient_id}_{nodule_id_only}_{slice_id_only}"

                # Find corresponding masks
                current_slice_mask_paths = []
                mask_parent_dirs = sorted(glob.glob(os.path.join(nodule_dir_path, "mask-*")))
                for mask_parent_dir_path in mask_parent_dirs:
                    mask_file_path = os.path.join(mask_parent_dir_path, slice_basename)
                    if os.path.exists(mask_file_path):
                        current_slice_mask_paths.append(mask_file_path)

                if current_slice_mask_paths: # Only proceed if masks are found for this slice
                    all_image_paths.append(slice_image_file_path)
                    all_mask_paths_lists.append(current_slice_mask_paths)

                    # Get text annotation for this image_id
                    # text = text_dict.get(current_image_id, "No description available") # Default text
                    # all_text_annotations.append(text)
                    text = text_dict.get(patient_id, "No description available") # Default text
                    all_text_annotations.append(text)
                else:
                    logger.log(f"Warning: No masks found for image {slice_image_file_path} (ID: {current_image_id})")

    if not all_image_paths:
        logger.log("ERROR: No image and mask pairs found. Check data_dir and CSV image_ids.")
        sys.exit(1)

    logger.log(f"Found {len(all_image_paths)} image/mask groups for training.")

    # Initialize text encoder
    text_encoder = TextEncoder(
        model_name=args.text_model_name,
        max_length=args.max_text_length,
        output_dim=args.text_dim
    )

    # Freeze text encoder for efficiency
    if args.freeze_text_encoder:
        text_encoder.freeze()

    # Create dataset
    dataset = TextGuidedImageSegDataset(
        image_paths=all_image_paths,         # Use the new list
        mask_paths=all_mask_paths_lists,     # Use the new list of lists
        text_annotations=all_text_annotations, # Use the new list
        image_size=args.image_size,
        class_cond=args.class_cond,
        return_dict=True,
        image_channels=args.image_actual_channels, # Pass the argument here
    )

    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    logger.log(f"DataLoader created with {len(loader.dataset)} samples.")

    # Create model and diffusion
    model_config_script_util = args_to_dict(args, model_and_diffusion_defaults().keys())

    # Add/override specific text guidance parameters from TextDiff UNet
    model_config_script_util["use_text_guidance"] = True # This should be handled by TextGuidedUNetModel directly
    model_config_script_util["text_dim"] = args.text_dim   # This should be handled by TextGuidedUNetModel directly

    logger.log(f"Creating model with args: {model_config_script_util}")
    model, diffusion, prior, posterior = create_model_and_diffusion(
        **model_config_script_util, # Pass all relevant args
    )
    
    # Convert prior and posterior to the correct dtype (float32 in this case, as use_fp16 is False)
    # This ensures their parameters match the expected input type.
    if prior is not None:
        prior = prior.float()
    if posterior is not None:
        posterior = posterior.float()

    # Move to GPU
    model.to(dist_util.dev())
    text_encoder.to(dist_util.dev())
    if prior is not None: # Check if prior exists before moving
        prior.to(dist_util.dev())       
    if posterior is not None: # Check if posterior exists before moving
        posterior.to(dist_util.dev())

    # Create schedule sampler
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion, diffusion.num_timesteps)

    # Start training
    logger.log("Starting training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=loader,
        prior=prior,             
        posterior=posterior,       
        classifier=None,         
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        text_encoder=text_encoder,  # Pass text encoder to train loop
        grad_clip=1.0,  # A value of 1.0 is a good starting point
        disable_text_guidance=args.disable_text_guidance,
    ).run_loop()

def create_argparser():
    defaults = dict(
        data_dir="",
        text_annotations_file="lidc_annotations_text.csv",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=4,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        text_model_name="emilyalsentzer/Bio_ClinicalBERT",
        max_text_length=128,
        text_dim=768,
        freeze_text_encoder=True,
        num_workers=4,
        log_dir="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("--disable_text_guidance", action="store_true", help="Disable text guidance for ablation")
    return parser

if __name__ == "__main__":
    main()
