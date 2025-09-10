"""Entry point for training the fMRI-to-CLIP model (cosmetic docs only)."""

import torch
import utils
from NSDdataset import NSDAccess
from config import args
from trainer import Trainer_Multi
from models import fMRI2CLIP, Clipper


def setup_environment():
    """Set CUDA flags and seed everything; return active device."""
    torch.backends.cuda.matmul.allow_tf32 = True
    utils.seed_everything(args.seed, cudnn_deterministic=False)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_clip_model(computing_device):
    """Instantiate CLIP wrapper according to config on the desired device."""
    if not args.norm_embs:
        print("WARNING: YOU WANT NORMED EMBEDDINGS FOR VERSATILE DIFFUSION!")
    model = Clipper(
        args.clip_variant,
        device=computing_device,
        hidden_state=True,
        norm_embs=args.norm_embs
    )
    print("clip model initialized.")
    return model

def initialize_brain_model(computing_device):
    """Create the fMRI2CLIP model with subject-dimension metadata."""
    brain_dimensions = {
        'subj1': 15724,
        'subj2': 14278,
        'subj5': 13039,
        'subj7': 12682
    }
    
    brain_model = fMRI2CLIP(
        subject_dims=brain_dimensions,
        d_model=768,
        fmri_seq_len=100,
        image_seq_len=257,
        text_seq_len=77,
        num_experts=16,
        slots_per_expert=4
    ).to(computing_device)
    
    print("brain model initialized.")
    print("parameter count:")
    utils.count_params(brain_model)
    
    return brain_model

def load_text_data(data_path):
    """Read COCO captions via NSD access helper for a fixed range of indices."""
    data_accessor = NSDAccess(data_path)
    sample_indices = list(range(73000))
    captions = data_accessor.read_image_coco_info(sample_indices, info_type='captions')
    print("text data loaded.")
    return captions

def initialize_training_system(config, brain_model, clip_model, text_data, computing_device):
    """Bundle models and data into a training controller instance."""
    return Trainer_Multi(config, brain_model, clip_model, text_data, computing_device)

def execute():
    """Full program execution: setup, initialize, load data, and train."""
    computing_device = setup_environment()
    
    # Step 1: Initialize models
    clip_model = initialize_clip_model(computing_device)
    brain_model = initialize_brain_model(computing_device)
    
    # Step 2: Load data
    text_data = load_text_data(args.data_path)
    
    # Step 3: Setup training
    training_system = initialize_training_system(
        args, brain_model, clip_model, text_data, computing_device
    )
    
    # Step 4: Execute training
    #training_system.prepare_wandb(args)
    training_system.train()
    
    print("\n===Execution Complete!===\n")

if __name__ == '__main__':
    execute()