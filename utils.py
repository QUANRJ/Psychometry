import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import PIL
import random
import os
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def torch_to_Image(x):
    if x.ndim==4:
        x=x[0]
    return transforms.ToPILImage()(x)

def np_to_Image(x):
    if x.ndim==4:
        x=x[0]
    return PIL.Image.fromarray((x.transpose(1, 2, 0)*127.5+128).clip(0,255).astype('uint8'))

def Image_to_torch(x):
    try:
        x = (transforms.ToTensor()(x)[:3].unsqueeze(0)-.5)/.5
    except:
        x = (transforms.ToTensor()(x[0])[:3].unsqueeze(0)-.5)/.5
    return x

def torch_to_matplotlib(x,device=device):
    if torch.mean(x)>10:
        x = (x.permute(0, 2, 3, 1)).clamp(0, 255).to(torch.uint8)
    else:
        x = (x.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)
    if device=='cpu':
        return x[0]
    else:
        return x.cpu().numpy()[0]

def pairwise_cosine_similarity(A, B, dim=1, eps=1e-8):
    numerator = A @ B.T
    A_l2 = torch.mul(A, A).sum(axis=dim)
    B_l2 = torch.mul(B, B).sum(axis=dim)
    denominator = torch.max(torch.sqrt(torch.outer(A_l2, B_l2)), torch.tensor(eps))
    return torch.div(numerator, denominator)

def batchwise_cosine_similarity(Z,B):
    B = B.T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
    return cosine_similarity

def batchwise_cosine_similarity_T(Z,B):
    B = B.T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm))
    return cosine_similarity

def topk(similarities,labels,k=5):
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum=0
    for i in range(k):
        topsum += torch.sum(torch.argsort(similarities,axis=1)[:,-(i+1)] == labels)/len(labels)
    return topsum

def soft_clip_loss(preds, targs, temp=0.005, eps=1e-10):
    clip_clip = (targs @ targs.T)/temp + eps
    check_loss(clip_clip, "clip_clip")
    brain_clip = (preds @ targs.T)/temp + eps
    check_loss(brain_clip, "brain_clip")
    
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    check_loss(loss1, "loss1")
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    check_loss(loss2, "loss2")
    
    loss = (loss1 + loss2)/2
    return loss

def gather_features(image_features, voxel_features):  
    all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
    if voxel_features is not None:
        all_voxel_features = torch.cat(torch.distributed.nn.all_gather(voxel_features), dim=0)
        return all_image_features, all_voxel_features
    return all_image_features

def soft_cont_loss(student_preds, teacher_preds, teacher_aug_preds, temp=0.125, distributed=False):
    if not distributed:
        teacher_teacher_aug = (teacher_preds @ teacher_aug_preds.T)/temp
        teacher_teacher_aug_t = (teacher_aug_preds @ teacher_preds.T)/temp
        student_teacher_aug = (student_preds @ teacher_aug_preds.T)/temp
        student_teacher_aug_t = (teacher_aug_preds @ student_preds.T)/temp
    else:
        all_student_preds, all_teacher_preds = gather_features(student_preds, teacher_preds)
        all_teacher_aug_preds = gather_features(teacher_aug_preds, None)

        teacher_teacher_aug = (teacher_preds @ all_teacher_aug_preds.T)/temp
        teacher_teacher_aug_t = (teacher_aug_preds @ all_teacher_preds.T)/temp
        student_teacher_aug = (student_preds @ all_teacher_aug_preds.T)/temp
        student_teacher_aug_t = (teacher_aug_preds @ all_student_preds.T)/temp
    
    loss1 = -(student_teacher_aug.log_softmax(-1) * teacher_teacher_aug.softmax(-1)).sum(-1).mean()
    loss2 = -(student_teacher_aug_t.log_softmax(-1) * teacher_teacher_aug_t.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('param counts:\n{:,} total\n{:,} trainable'.format(total, trainable))

def image_grid(imgs, rows, cols):
    w, h = imgs[0].size
    grid = PIL.Image.new('RGB', size=(cols*w, rows*h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def check_loss(loss, message="loss"):
    if loss.isnan().any():
        raise ValueError(f'NaN loss in {message}')

def cosine_anneal(start, end, steps):
    return end + (start - end)/2 * (1 + torch.cos(torch.pi*torch.arange(steps)/(steps-1)))

def decode_latents(latents,vae):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return image

def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        print('Note: not using cudnn.deterministic')


def Ecphory(
        pred_embedding_vision, pred_embedding_text,
        pred_image_train_norm, pred_text_train_norm,
        clip_vision_train, clip_text_train,
        clip_image_target_norm, clip_text_target_norm,
        alpha, recons_per_sample
        ):
    pred_embedding_text = pred_embedding_text.to(pred_text_train_norm.device)
    clip_image_target_norm = clip_image_target_norm.to(pred_text_train_norm.device)
    pred_embedding_vision = pred_embedding_vision.to(pred_text_train_norm.device)
    clip_text_target_norm = clip_text_target_norm.to(pred_text_train_norm.device)
    pred_embedding_vision_norm = nn.functional.normalize(pred_embedding_vision.flatten(1), dim=-1)
    pred_embedding_text_norm = nn.functional.normalize(pred_embedding_text.flatten(1), dim=-1)      
    similarity_text_tar_with_train = batchwise_cosine_similarity(pred_embedding_text_norm, pred_text_train_norm)
    similarity_vision_tar_with_train = batchwise_cosine_similarity(pred_embedding_vision_norm, pred_image_train_norm) 
    topk_index_text_tar_with_train = torch.topk(similarity_text_tar_with_train.flatten(), recons_per_sample).indices
    topk_index_vision_tar_with_train = torch.topk(similarity_vision_tar_with_train.flatten(), recons_per_sample).indices

    combined_brain_clip_text_embeddings_using_target = (1-alpha) * clip_text_train[topk_index_text_tar_with_train] + alpha * pred_embedding_text
    combined_brain_clip_image_embeddings_using_target = (1-alpha) * clip_vision_train[topk_index_vision_tar_with_train] + alpha * pred_embedding_vision
  
    return combined_brain_clip_image_embeddings_using_target, combined_brain_clip_text_embeddings_using_target




@torch.no_grad()
def reconstruction_moe_Ecphory(
    subj, image, captions, voxel, clip_image_train, clip_text_train, pred_image_train, pred_text_train, voxel2clip,
    clip_extractor,
    unet, vae, noise_scheduler,
    img_lowlevel=None,
    num_inference_steps=50,
    recons_per_sample=1,
    guidance_scale=7.5,
    img2img_strength=.85,
    seed=42,
    plotting=True,
    verbose=False,
    n_samples_save=1,
    device=None,
    mem_efficient=True,
    use_ecphory=True,
    alpha=0.5,
):
    if n_samples_save != 1:
        raise ValueError("n_samples_save must = 1. Function must be called one image at a time")
    if recons_per_sample <= 0:
        raise ValueError("recons_per_sample must > 0")

    vox = voxel[:n_samples_save]
    img = image[:n_samples_save]
    batch = vox.shape[0]

    if mem_efficient:
        for m in [clip_extractor, unet, vae]:
            m.to("cpu")
    else:
        for m in [clip_extractor, unet, vae]:
            m.to(device)

    if unet is not None:
        vae_scale = 2 ** (len(vae.config.block_out_channels) - 1)
        img_h = unet.config.sample_size * vae_scale
        img_w = unet.config.sample_size * vae_scale
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    tgt_img_feat = clip_extractor.embed_image(img)
    tgt_img_feat_norm = nn.functional.normalize(tgt_img_feat.flatten(1), dim=-1)
    tgt_txt_feat = clip_extractor.embed_text(captions)
    tgt_txt_feat_norm = nn.functional.normalize(tgt_txt_feat.flatten(1), dim=-1)
    pred_img_train_norm = nn.functional.normalize(pred_image_train.flatten(1), dim=-1)
    pred_txt_train_norm = nn.functional.normalize(pred_text_train.flatten(1), dim=-1)

    if voxel2clip is not None:
        if subj is not None:
            brain_feats = voxel2clip(vox, f'subj{subj.item()}')
        else:
            brain_feats = voxel2clip(vox)
        if mem_efficient:
            voxel2clip.to('cpu')
        brain_img_emb, brain_txt_emb = brain_feats[:2]
        brain_img_emb, brain_txt_emb = Ecphory(
            brain_img_emb, brain_txt_emb,
            pred_img_train_norm, pred_txt_train_norm,
            clip_image_train, clip_text_train,
            tgt_img_feat_norm, tgt_txt_feat_norm, alpha, recons_per_sample)
    else:
        brain_img_emb = None
        brain_txt_emb = None

    for idx in range(len(brain_img_emb)):
        norm_img = brain_img_emb[idx, 0].norm(dim=-1).reshape(-1, 1, 1) + 1e-6
        norm_txt = brain_txt_emb[idx, 0].norm(dim=-1).reshape(-1, 1, 1) + 1e-6
        brain_img_emb[idx] = brain_img_emb[idx] / norm_img
        brain_txt_emb[idx] = brain_txt_emb[idx] / norm_txt

    latent_input = brain_img_emb
    prompt_input = brain_txt_emb
    if verbose:
        print("latent_input", latent_input.shape)
        print("prompt_input", prompt_input.shape)
    do_guidance = guidance_scale > 1.0
    if do_guidance:
        latent_input = torch.cat([torch.zeros_like(latent_input), latent_input]).to(device).to(unet.dtype)
        prompt_input = torch.cat([torch.zeros_like(prompt_input), prompt_input]).to(device).to(unet.dtype)
    model_input = torch.cat([prompt_input, latent_input], dim=1)
    noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)
    bsz = model_input.shape[0] // 2
    shape = (bsz, unet.in_channels, img_h // vae_scale, img_w // vae_scale)

    if img_lowlevel is not None:
        init_step = min(int(num_inference_steps * img2img_strength), num_inference_steps)
        t0 = max(num_inference_steps - init_step, 0)
        t_seq = noise_scheduler.timesteps[t0:]
        latent_t = t_seq[:1].repeat(bsz)
        if verbose:
            print("img_lowlevel", img_lowlevel.shape)
        lowlevel_emb = clip_extractor.normalize(img_lowlevel)
        if mem_efficient:
            vae.to(device)
        init_lat = vae.encode(lowlevel_emb.to(device).to(vae.dtype)).latent_dist.sample(gen)
        init_lat = vae.config.scaling_factor * init_lat
        init_lat = init_lat.repeat(recons_per_sample, 1, 1, 1)
        noise = torch.randn([recons_per_sample, 4, 64, 64], device=device, generator=gen, dtype=model_input.dtype)
        init_lat = noise_scheduler.add_noise(init_lat, noise, latent_t)
        latents = init_lat
        t_seq_run = t_seq
    else:
        t_seq_run = noise_scheduler.timesteps
        latents = torch.randn([recons_per_sample, 4, 64, 64], device=device, generator=gen, dtype=model_input.dtype)
        latents = latents * noise_scheduler.init_noise_sigma

    if mem_efficient:
        unet.to(device)
    for idx, t in enumerate(t_seq_run):
        model_latent = torch.cat([latents] * 2) if do_guidance else latents
        model_latent = noise_scheduler.scale_model_input(model_latent, t)
        if verbose:
            print(f"step {idx}, model_latent: {model_latent.shape}, model_input: {model_input.shape}")
        noise_pred = unet(model_latent, t, encoder_hidden_states=model_input).sample
        if do_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
    if mem_efficient:
        unet.to("cpu")
    recons = decode_latents(latents.to(device), vae.to(device)).detach().cpu()
    brain_recons = recons.unsqueeze(0)

    if verbose:
        print("brain_recons", brain_recons.shape)

    best_indices = np.zeros(n_samples_save, dtype=np.int16)
    if mem_efficient:
        vae.to("cpu")
        unet.to("cpu")
        clip_extractor.to(device)
    sim_scores = []
    for i in range(recons_per_sample):
        rec_feat = clip_extractor.embed_image(brain_recons[0, [i]].float()).to(tgt_img_feat_norm.device).to(tgt_img_feat_norm.dtype)
        rec_feat = nn.functional.normalize(rec_feat.view(len(rec_feat), -1), dim=-1)
        sim = batchwise_cosine_similarity(tgt_img_feat_norm, rec_feat)
        sim_scores.append(sim.item())
    if verbose:
        print(sim_scores)
    best_indices[0] = int(np.nanargmax(sim_scores))
    if verbose:
        print(best_indices)
    if mem_efficient:
        clip_extractor.to("cpu")
        voxel2clip.to(device)

    img2img_flag = 0 if img_lowlevel is None else 1
    ncols = 1 + img2img_flag + recons_per_sample
    if plotting:
        fig, axes = plt.subplots(n_samples_save, ncols, figsize=(ncols * 5, 6 * n_samples_save), facecolor=(1, 1, 1))
    else:
        fig = None
        recon_img = None
    im_idx = 0
    if plotting:
        axes[0].set_title("Original Image")
        axes[0].imshow(torch_to_Image(img[im_idx]))
        if img2img_flag == 1:
            axes[1].set_title(f"Img2img ({img2img_strength})")
            axes[1].imshow(torch_to_Image(img_lowlevel[im_idx].clamp(0, 1)))
    for j, k in enumerate(range(ncols - recons_per_sample, ncols)):
        recon = brain_recons[im_idx][j]
        if plotting:
            if j == best_indices[im_idx]:
                axes[k].set_title("Reconstruction", fontweight='bold')
                recon_img = recon
            else:
                axes[k].set_title(f"Recon {j+1} from brain")
            axes[k].imshow(torch_to_Image(recon))
    if plotting:
        for ax in axes if isinstance(axes, np.ndarray) else [axes]:
            if isinstance(ax, np.ndarray):
                for a in ax:
                    a.axis('off')
            else:
                ax.axis('off')
    return fig, brain_recons, best_indices, recon_img