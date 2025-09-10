import clip
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.nn import TransformerDecoderLayer
import torch.nn.functional as F
from typing import List, Optional
from diffusers.models.vae import Decoder


class Clipper(torch.nn.Module):
    """Thin wrapper around CLIP encoders with optional hidden-state outputs."""
    def __init__(self, clip_variant, clamp_embs=False, norm_embs=False,
                 hidden_state=False, device=torch.device('cpu')):
        super().__init__()
        assert clip_variant in ("RN50", "ViT-L/14", "ViT-B/32", "RN50x64"), \
            "clip_variant must be one of RN50, ViT-L/14, ViT-B/32, RN50x64"
        print(clip_variant, device)
        
        if clip_variant=="ViT-L/14" and hidden_state:
            from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection, CLIPTokenizer
            image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").eval()
            image_encoder = image_encoder.to(device)
            for param in image_encoder.parameters():
                param.requires_grad = False # dont need to calculate gradients
            self.image_encoder = image_encoder

            text_encoder = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").eval()
            text_encoder = text_encoder.to(device)
            for param in text_encoder.parameters():
                param.requires_grad = False # dont need to calculate gradients
            self.text_encoder = text_encoder
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        elif hidden_state:
            raise Exception("hidden_state embeddings only works with ViT-L/14 right now")
        
        clip_model, preprocess = clip.load(clip_variant, device=device)
        clip_model.eval() # dont want to train model
        for param in clip_model.parameters():
            param.requires_grad = False # dont need to calculate gradients
            
        self.clip = clip_model
        self.clip_variant = clip_variant
        if clip_variant == "RN50x64":
            self.clip_size = (448,448)
        else:
            self.clip_size = (224,224)
            
        preproc = transforms.Compose([
            transforms.Resize(size=self.clip_size[0], interpolation=transforms.InterpolationMode.BICUBIC, antialias=None),
            transforms.CenterCrop(size=self.clip_size),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        self.preprocess = preproc
        self.hidden_state = hidden_state
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073])
        self.std = np.array([0.26862954, 0.26130258, 0.27577711])
        self.normalize = transforms.Normalize(self.mean, self.std)
        self.denormalize = transforms.Normalize((-self.mean / self.std).tolist(), (1.0 / self.std).tolist())
        self.clamp_embs = clamp_embs
        self.norm_embs = norm_embs
        self.device= device
        
        def versatile_normalize_embeddings(encoder_output):
            embeds = encoder_output.last_hidden_state
            embeds = image_encoder.vision_model.post_layernorm(embeds)
            embeds = image_encoder.visual_projection(embeds)
            return embeds
        self.versatile_normalize_embeddings = versatile_normalize_embeddings

    def resize_image(self, image):
        """Resize tensor image to CLIP input resolution."""
        # note: antialias should be False if planning to use Pinkney's Image Variation SD model
        return transforms.Resize(self.clip_size, antialias=None)(image.to(self.device))

    def embed_image(self, image):
        """Encode images to CLIP image embeddings.

        Expects images normalized to [-1, 1].
        """
        if self.hidden_state:
            clip_emb = self.preprocess((image).to(self.device))
            clip_emb = self.image_encoder(clip_emb)
            clip_emb = self.versatile_normalize_embeddings(clip_emb)
        else:
            clip_emb = self.preprocess(image.to(self.device))
            clip_emb = self.clip.encode_image(clip_emb)
        if self.clamp_embs:
            clip_emb = torch.clamp(clip_emb, -1.5, 1.5)
        if self.norm_embs:
            if self.hidden_state:        
                clip_emb = clip_emb / torch.norm(clip_emb[:, 0], dim=-1).reshape(-1, 1, 1)
            else:
                clip_emb = nn.functional.normalize(clip_emb, dim=-1)
        return clip_emb
    
    def embed_text(self, prompt):
        r"""Encode text prompt(s) to CLIP text token embeddings.

        Args:
            prompt: A string or list of strings.
        Returns:
            Text embeddings aligned with CLIP vision space.
        """

        def normalize_embeddings(encoder_output):
            embeds = self.text_encoder.text_projection(encoder_output.last_hidden_state)
            embeds_pooled = encoder_output.text_embeds
            embeds = embeds / torch.norm(embeds_pooled.unsqueeze(1), dim=-1, keepdim=True)
            return embeds

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="max_length", return_tensors="pt").input_ids
        with torch.no_grad():
            prompt_embeds = self.text_encoder(
                text_input_ids.to(self.device),
            )
        prompt_embeds = normalize_embeddings(prompt_embeds)

        return prompt_embeds

    def embed_curated_annotations(self, annots):
        """Sample curated captions and encode them with the text encoder."""
        for i,b in enumerate(annots):
            t = ''
            while t == '':
                rand = torch.randint(5,(1,1))[0][0]
                t = b[0,rand]
            if i==0:
                txt = np.array(t)
            else:
                txt = np.vstack((txt,t))
        txt = txt.flatten()
        return self.embed_text(txt)



from diffusers.models.vae import Decoder
class Voxel2StableDiffusionModel(torch.nn.Module):
    """Map voxel vectors to Stable Diffusion latents via MLP + Decoder."""
    def __init__(self, in_dim=15724, h=4096, n_blocks=4, use_cont=False, ups_mode='4x'):
        super().__init__()
        self.lin0 = nn.Sequential(
            nn.Linear(in_dim, h, bias=False),
            nn.LayerNorm(h),
            nn.SiLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h, bias=False),
                nn.LayerNorm(h),
                nn.SiLU(inplace=True),
                nn.Dropout(0.25)
            ) for _ in range(n_blocks)
        ])
        self.ups_mode = ups_mode
        if ups_mode=='4x':
            self.lin1 = nn.Linear(h, 16384, bias=False)
            self.norm = nn.GroupNorm(1, 64)
            
            self.upsampler = Decoder(
                in_channels=64,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
                block_out_channels=[64, 128, 256],
                layers_per_block=1,
            )

            if use_cont:
                self.maps_projector = nn.Sequential(
                    nn.Conv2d(64, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=True),
                )
            else:
                self.maps_projector = nn.Identity()
        
        if ups_mode=='8x':  # prev best
            self.lin1 = nn.Linear(h, 16384, bias=False)
            self.norm = nn.GroupNorm(1, 256)
            
            self.upsampler = Decoder(
                in_channels=256,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
                block_out_channels=[64, 128, 256, 256],
                layers_per_block=1,
            )
            self.maps_projector = nn.Identity()
        
        if ups_mode=='16x':
            self.lin1 = nn.Linear(h, 8192, bias=False)
            self.norm = nn.GroupNorm(1, 512)
            
            self.upsampler = Decoder(
                in_channels=512,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D", "UpDecoderBlock2D"],
                block_out_channels=[64, 128, 256, 256, 512],
                layers_per_block=1,
            )
            self.maps_projector = nn.Identity()

            if use_cont:
                self.maps_projector = nn.Sequential(
                    nn.Conv2d(64, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=True),
                )
            else:
                self.maps_projector = nn.Identity()

    # @torchsnooper.snoop()
    def forward(self, x, return_transformer_feats=False):
        """Forward pass: returns decoded latents (and optional maps)."""
        x = self.lin0(x)
        residual = x
        for res_block in self.mlp:
            x = res_block(x)
            x = x + residual
            residual = x
        x = x.reshape(len(x), -1)
        x = self.lin1(x)  # bs, 4096

        if self.ups_mode == '4x':
            side = 16
        if self.ups_mode == '8x':
            side = 8
        if self.ups_mode == '16x':
            side = 4
        
        # decoder
        x = self.norm(x.reshape(x.shape[0], -1, side, side).contiguous())
        if return_transformer_feats:
            return self.upsampler(x), self.maps_projector(x).flatten(2).permute(0,2,1)
        return self.upsampler(x)
    

class LowlevelModel(torch.nn.Module):
    """Subject-specific projection to hidden space, then upsample to latents."""
    def __init__(self, subject_dims: dict, hidden_dims: list = [4096, 2048], h=4096, use_cont=True, ups_mode='4x'):
        super().__init__()

        self.subject_projs = nn.ModuleDict()
        #self.seq_len = seq_len
        
        for subj, dim in subject_dims.items():
            layers = []
            current_dim = dim
            for h_dim in hidden_dims:
                layers.extend([
                    nn.Linear(current_dim, h_dim),
                    nn.GELU()
                ])
                current_dim = h_dim
            layers.append(nn.Linear(current_dim, h))
            self.subject_projs[subj] = nn.Sequential(*layers)
       
        self.ups_mode = ups_mode
        if ups_mode=='4x':
            self.lin1 = nn.Linear(h, 16384, bias=False)
            self.norm = nn.GroupNorm(1, 64)
            
            self.upsampler = Decoder(
                in_channels=64,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
                block_out_channels=[64, 128, 256],
                layers_per_block=1,
            )

            if use_cont:
                self.maps_projector = nn.Sequential(
                    nn.Conv2d(64, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=True),
                )
            else:
                self.maps_projector = nn.Identity()
        
        if ups_mode=='8x':  # prev best
            self.lin1 = nn.Linear(h, 16384, bias=False)
            self.norm = nn.GroupNorm(1, 256)
            
            self.upsampler = Decoder(
                in_channels=256,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
                block_out_channels=[64, 128, 256, 256],
                layers_per_block=1,
            )
            self.maps_projector = nn.Identity()
        
        if ups_mode=='16x':
            self.lin1 = nn.Linear(h, 8192, bias=False)
            self.norm = nn.GroupNorm(1, 512)
            
            self.upsampler = Decoder(
                in_channels=512,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D", "UpDecoderBlock2D"],
                block_out_channels=[64, 128, 256, 256, 512],
                layers_per_block=1,
            )
            self.maps_projector = nn.Identity()

            if use_cont:
                self.maps_projector = nn.Sequential(
                    nn.Conv2d(64, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=True),
                )
            else:
                self.maps_projector = nn.Identity()

    # @torchsnooper.snoop()
    def forward(self, x: torch.Tensor, subject_id: str, return_transformer_feats=False):
        """Forward pass for a given subject identifier."""
        proj = self.subject_projs[subject_id]
        x = proj(x)
        #x = x.reshape(len(x), -1)
        x = self.lin1(x)  # bs, 4096

        if self.ups_mode == '4x':
            side = 16
        if self.ups_mode == '8x':
            side = 8
        if self.ups_mode == '16x':
            side = 4
        
        # decoder
        x = self.norm(x.reshape(x.shape[0], -1, side, side).contiguous())
        if return_transformer_feats:
            return self.upsampler(x), self.maps_projector(x).flatten(2).permute(0,2,1)
        return self.upsampler(x)
  

class ExpertMLP(nn.Module):
    """Simple two-layer MLP used inside MoE blocks."""
    def __init__(self, d_model: int, dim_feedforward: int, activation: str = 'gelu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = F.gelu if activation == 'gelu' else F.relu
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear2.weight)

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """Apply MLP to inputs; task_id kept for extensibility."""
        return self.linear2(self.activation(self.linear1(x)))

class SoftMoELayer(nn.Module):
    """Soft mixture-of-experts with subject-specific gating tensors."""
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        slots_per_expert: int,
        all_subjects: List[str],
        dim_feedforward: int = 3072,
        activation: str = 'gelu',
        normalize: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.normalize = normalize
        
        self.experts = nn.ModuleList([
            ExpertMLP(d_model, dim_feedforward, activation)
            for _ in range(num_experts)
        ])
        
        self.gates = nn.ParameterDict({
            subj: nn.Parameter(torch.zeros(d_model, num_experts, slots_per_expert))
            for subj in all_subjects
        })
        
        for subj in all_subjects:
            nn.init.normal_(self.gates[subj], mean=0, std=1/d_model**0.5)
        
        self.scale = nn.Parameter(torch.ones(1)) if normalize else None

    def forward(self, x: torch.Tensor, subject_id: str) -> torch.Tensor:
        """Route tokens through experts and recombine via learned gates."""
        batch_size, seq_len, _ = x.shape
        
        if self.normalize:
            x = F.normalize(x, dim=-1)
            gate = self.scale * F.normalize(self.gates[subject_id], dim=0)
        else:
            gate = self.gates[subject_id]

        logits = torch.einsum('bsd,dnp->bsnp', x, gate)
        dispatch = F.softmax(logits, dim=1)
        combine_shape = logits.shape
        combine_logits = logits.reshape(batch_size, seq_len, -1) 
        combine_weights = F.softmax(combine_logits, dim=-1)
        combine = combine_weights.view(combine_shape) 

        expert_inputs = torch.einsum('bsd,bsnp->bnpd', x, dispatch)
        expert_outputs = []
        
        for expert_idx, expert in enumerate(self.experts):
            expert_out = expert(
                expert_inputs[:, expert_idx],
                task_id=expert_idx 
            )
            expert_outputs.append(expert_out)
        
        combined = torch.stack(expert_outputs, dim=1)
        return torch.einsum('bnpd,bsnp->bsd', combined, combine)

class MoEDecoderLayer(nn.Module):
    """Transformer decoder layer with self/cross-attn and SoftMoE feedforward."""
    def __init__(
        self,
        d_model: int,
        nhead: int,
        all_subjects: List[str],
        num_experts: int = 4,
        slots_per_expert: int = 4,
        dim_feedforward: int = 3072,
        activation: str = 'gelu'
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead)
        
        self.moe = SoftMoELayer(
            d_model=d_model,
            num_experts=num_experts,
            slots_per_expert=slots_per_expert,
            all_subjects=all_subjects,
            dim_feedforward=dim_feedforward,
            activation=activation
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        subject_id: str
    ) -> torch.Tensor:
        """Run SA -> CA -> MoE with residuals and layer norms."""
        attn_out, _ = self.self_attn(tgt, tgt, tgt)
        tgt = tgt + self.dropout(attn_out)
        tgt = self.norm1(tgt)
        
        cross_out, _ = self.cross_attn(tgt, memory, memory)
        tgt = tgt + self.dropout(cross_out)
        tgt = self.norm2(tgt)
        
        tgt_transposed = tgt.permute(1, 0, 2)  # [seq_len, batch, dim] -> [batch, seq_len, dim]
        moe_out = self.moe(tgt_transposed, subject_id)
        moe_out = moe_out.permute(1, 0, 2)  
        
        tgt = tgt + self.dropout(moe_out)
        return self.norm3(tgt)

class fMRI2CLIP(nn.Module):
    """Encode fMRI signals and decode into CLIP-aligned image/text tokens."""
    def __init__(
        self,
        subject_dims: dict,
        d_model: int = 768,
        fmri_seq_len: int = 50,
        image_seq_len: int = 257,
        text_seq_len: int = 77,
        num_experts: int = 4,
        slots_per_expert: int = 4,
        proj_hidden_dims: list = [4096, 2048]
    ):
        super().__init__()
        self.all_subjects = list(subject_dims.keys())

        self.fMRI_encoder = fMRIEncoder(
            subject_dims=subject_dims,
            d_model=d_model,
            seq_len=fmri_seq_len,
            hidden_dims=proj_hidden_dims
        )
        

        self.image_decoder = self.build_moe_decoder(
            num_tokens=image_seq_len,
            d_model=d_model,
            num_experts=num_experts,
            slots_per_expert=slots_per_expert,
            all_subjects=self.all_subjects
        )
        self.text_decoder = self.build_moe_decoder(
            num_tokens=text_seq_len,
            d_model=d_model,
            num_experts=num_experts,
            slots_per_expert=slots_per_expert,
            all_subjects=self.all_subjects
        )
        
        self.image_proj = nn.Linear(d_model, d_model)
        self.text_proj = nn.Linear(d_model, d_model)

    def build_moe_decoder(self, num_tokens: int, d_model: int, num_experts: int,
                         slots_per_expert: int, all_subjects: List[str]) -> nn.Module:
        class CustomDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.query = nn.Parameter(torch.randn(num_tokens, d_model))
                self.layers = nn.ModuleList([
                    TransformerDecoderLayer(d_model, 8, 3072, activation='gelu')
                    for _ in range(3)
                ])
                self.moe_layer = MoEDecoderLayer(
                    d_model=d_model,
                    nhead=8,
                    all_subjects=all_subjects,
                    num_experts=num_experts,
                    slots_per_expert=slots_per_expert
                )
            
            def forward(self, src: torch.Tensor, subject_id: str) -> torch.Tensor:
                """Decode features into token embeddings using transformer + MoE."""
                memory = src.permute(1, 0, 2)
                tgt = self.query.unsqueeze(1).repeat(1, src.size(0), 1)
                
                for layer in self.layers:
                    tgt = layer(tgt, memory)
                
                return self.moe_layer(tgt, memory, subject_id).permute(1, 0, 2)
        
        return CustomDecoder()


    def forward(self, x: torch.Tensor, subject_id: str) -> tuple:
        """Return (image_embeddings, text_embeddings) for a subject input."""
        features = self.fMRI_encoder(x, subject_id)
        
        image_emb = self.image_proj(self.image_decoder(features, subject_id))
        text_emb = self.text_proj(self.text_decoder(features, subject_id))
        
        return image_emb, text_emb

class fMRIEncoder(nn.Module):
    """Project per-subject fMRI vectors to a shared token sequence with PE."""
    def __init__(
        self,
        subject_dims: dict,  
        d_model: int = 768,
        seq_len: int = 50,
        hidden_dims: list = [4096, 2048], 
    ):
        super().__init__()
        self.subject_projs = nn.ModuleDict()
        self.d_model = d_model
        self.seq_len = seq_len
        
        for subj, dim in subject_dims.items():
            layers = []
            current_dim = dim
            for h_dim in hidden_dims:
                layers.extend([
                    nn.Linear(current_dim, h_dim),
                    nn.GELU()
                ])
                current_dim = h_dim
            layers.append(nn.Linear(current_dim, seq_len*d_model))
            self.subject_projs[subj] = nn.Sequential(*layers)
        
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model))

    def forward(self, x: torch.Tensor, subject_id: str) -> torch.Tensor:
        """Return token sequence for subject, with positional embedding added."""

        proj = self.subject_projs[subject_id]
        x = proj(x)  # [batch, seq_len*d_model]
        x = x.view(x.size(0), self.seq_len, self.d_model)
        return x + self.pos_embed

