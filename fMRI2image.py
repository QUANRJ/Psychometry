import os
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm

import utils
import data
from config import args
from models import Clipper, Voxel2StableDiffusionModel, fMRI2CLIP
from NSDdataset import NSDAccess

def prepare_voxel2sd(args, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_state = checkpoint['model_state_dict']
    model = Voxel2StableDiffusionModel(in_dim=args.num_voxels)
    model.load_state_dict(model_state, strict=False)
    model.to(device)
    model.eval()
    print("Low-level model loaded.")
    return model

def prepare_data(args):
    voxel_counts = {1: 15724, 2: 14278, 3: 15226, 4: 13153, 5: 13039, 6: 17907, 7: 12682, 8: 14386}
    args.num_voxels = voxel_counts[args.subj_test]
    test_dir = f"{args.data_path}/webdataset_avg_split/test/subj0{args.subj_test}"
    loader = data.get_dataloader(
        test_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        is_shuffle=False,
        extensions=['nsdgeneral.npy', 'jpg', 'coco73k.npy', 'subj'],
        pool_type=args.pool_type,
        pool_num=args.pool_num,
    )
    return loader

def prepare_VD(args, device):
    print('Initializing Versatile Diffusion pipeline...')
    from diffusers import VersatileDiffusionDualGuidedPipeline, UniPCMultistepScheduler
    from diffusers.models import DualTransformer2DModel
    try:
        pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained(args.vd_cache_dir)
    except Exception:
        print(f"Downloading Versatile Diffusion to {args.vd_cache_dir}")
        pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained(
            "shi-labs/versatile-diffusion",
            cache_dir=args.vd_cache_dir
        )
    pipe.image_unet.eval().to(device)
    pipe.vae.eval().to(device)
    pipe.image_unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.scheduler = UniPCMultistepScheduler.from_pretrained(
        "shi-labs/versatile-diffusion", cache_dir=args.vd_cache_dir, subfolder="scheduler"
    )
    for n, m in pipe.image_unet.named_modules():
        if isinstance(m, DualTransformer2DModel):
            m.mix_ratio = args.text_image_ratio
            for idx, cond in enumerate(("text", "image")):
                if cond == "text":
                    m.condition_lengths[idx] = 77
                    m.transformer_index_for_condition[idx] = 1
                else:
                    m.condition_lengths[idx] = 257
                    m.transformer_index_for_condition[idx] = 0
    return pipe

def prepare_CLIP(args, device):
    clip_dims = {"RN50": 1024, "ViT-L/14": 768, "ViT-B/32": 512, "ViT-H-14": 1024}
    dim = clip_dims[args.clip_variant]
    img_dim = 257 * dim
    txt_dim = 77 * dim
    extractor = Clipper("ViT-L/14", hidden_state=True, norm_embs=True, device=device)
    return extractor, img_dim, txt_dim

def prepare_voxel2clip(args, out_dim_image, out_dim_text, device):
    subj_dims = {'subj1': 15724, 'subj2': 14278, 'subj5': 13039, 'subj7': 12682}
    model = fMRI2CLIP(
        subject_dims=subj_dims,
        d_model=768,
        fmri_seq_len=100,
        image_seq_len=257,
        text_seq_len=77,
        num_experts=16,
        slots_per_expert=4
    ).to(device)
    ckpt_dir = f'/data1/ruijie/train_logs/{args.model_name}'
    ckpt_file = os.path.join(ckpt_dir, f'{args.ckpt_from}.pth')
    print("Loading checkpoint:", ckpt_file)
    ckpt = torch.load(ckpt_file, map_location='cpu')
    print("Checkpoint epoch:", ckpt['epoch'])
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.requires_grad_(False)
    model.eval().to(device)
    return model

def prepare_coco(args):
    nsd = NSDAccess(args.data_path)
    idxs = list(range(73000))
    captions = nsd.read_image_coco_info(idxs, info_type='captions')
    print("COCO captions loaded.")
    return captions

def prepare_ecphory(args, output_dir):
    pred_img_path = os.path.join(output_dir, "all_pred_image_embeddings.pt")
    pred_txt_path = os.path.join(output_dir, "all_pred_text_embeddings.pt")
    clip_img_path = os.path.join(output_dir, "all_clip_image_embeddings.pt")
    clip_txt_path = os.path.join(output_dir, "all_clip_text_embeddings.pt")
    clip_img = torch.load(clip_img_path, map_location='cpu')
    clip_txt = torch.load(clip_txt_path, map_location='cpu')
    pred_img = torch.load(pred_img_path, map_location='cpu')
    pred_txt = torch.load(pred_txt_path, map_location='cpu')
    return clip_img, clip_txt, pred_img, pred_txt

def main(device):
    args.batch_size = 1
    if args.subj_load is None:
        args.subj_load = [args.subj_test]
    test_loader = prepare_data(args)
    coco_captions = prepare_coco(args)
    total_samples = len(test_loader)
    ae_dir = f'../train_logs/{args.autoencoder_name}'
    ae_ckpt = os.path.join(ae_dir, 'best.pth')
    if os.path.exists(ae_ckpt):
        voxel2sd_model = prepare_voxel2sd(args, ae_ckpt, device)
        args.pool_type = None
    else:
        print("No valid path for low-level model specified; not using img2img!")
        args.img2img_strength = 1
    vd_pipeline = prepare_VD(args, device)
    unet = vd_pipeline.image_unet
    vae = vd_pipeline.vae
    scheduler = vd_pipeline.scheduler
    clipper, img_dim, txt_dim = prepare_CLIP(args, device)
    voxel2clip_model = prepare_voxel2clip(args, img_dim, txt_dim, device)
    out_dir = f'/data1/ruijie/train_logs/{args.model_name}'
    save_path = os.path.join(out_dir, f"recon_on_subj_{args.subj_test}_{args.alpha}")
    os.makedirs(save_path, exist_ok=True)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    indices = np.arange(total_samples)
    # if args.test_end is None:
    #     args.test_end = total_samples
    only_low = False
    if args.img2img_strength == 1:
        use_img2img = False
    elif args.img2img_strength == 0:
        use_img2img = True
        only_low = True
    else:
        use_img2img = True
    clip_img_train, clip_txt_train, pred_img_train, pred_txt_train = prepare_ecphory(args, args.output_dir)
    for idx, (vox, image, coco, subj) in enumerate(tqdm(test_loader, total=len(indices))):
        # if idx < args.test_start:
        #     continue
        # if idx >= total_samples:
        #     break
        # if args.samples is not None and idx not in args.samples:
        #     continue
        vox = torch.mean(vox, axis=1).float().to(device)
        image = image.to(device)
        rep_idx = idx % 3
        coco_flat = coco.squeeze()
        if coco_flat.dim() == 0:
            coco_ids = [coco_flat.item()]
        else:
            coco_ids = coco_flat.tolist()
        prompt_list = [coco_captions[cid] for cid in coco_ids]
        captions = [p[rep_idx]['caption'] for p in prompt_list]
        with torch.no_grad():
            if args.only_embeddings:
                emb = voxel2clip_model(vox, f'subj{subj.item()}')
                torch.save(emb[:2], os.path.join(save_path, f'embeddings_{idx}.pt'))
                continue
            if use_img2img:
                ae_pred = voxel2sd_model(vox)
                blurry = vd_pipeline.vae.decode(ae_pred.to(device)/0.18215).sample / 2 + 0.5
                vox = data.pool_voxels(vox, args.pool_num, args.pool_type)
            else:
                blurry = None
            if only_low:
                recon = blurry
            else:
                grid, recon, picks, recon_img = utils.reconstruction_moe_Ecphory(
                    subj, image, captions, vox, clip_img_train, clip_txt_train, pred_img_train, pred_txt_train, voxel2clip_model,
                    clipper, unet, vae, scheduler,
                    img_lowlevel=blurry,
                    num_inference_steps=args.num_inference_steps,
                    n_samples_save=args.batch_size,
                    recons_per_sample=args.recons_per_sample,
                    guidance_scale=args.guidance_scale,
                    img2img_strength=args.img2img_strength,
                    seed=args.seed,
                    plotting=args.plotting,
                    verbose=args.verbose,
                    device=device,
                    mem_efficient=False,
                    use_ecphory=args.use_ecphory,
                    alpha=args.alpha,
                )
                if args.plotting:
                    grid.savefig(os.path.join(save_path, f'{idx}.png'))
                recon = recon[:, picks.astype(np.int8)]
                torch.save(image, os.path.join(save_path, f'{idx}_img.pt'))
                torch.save(recon, os.path.join(save_path, f'{idx}_rec.pt'))
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("Results saved to:", save_path)

if __name__ == "__main__":
    utils.seed_everything(seed=args.seed)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    main(device)
