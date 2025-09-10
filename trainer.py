"""Training loop and orchestration utilities (cosmetic docs only)."""

from abc import abstractmethod
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# Custom models and functions #
import utils
import data

class CoreHandler:
    """Base class encapsulating dataloaders, optimizers, and train/eval steps."""
    def __init__(self, cfg, net, extractor, prompt_bank, dev) -> None:
        self._logdir = os.path.abspath(f'../train_logs/{cfg.model_name}')
        if not os.path.exists(self._logdir):
            os.makedirs(self._logdir, exist_ok=True)
        self.cfg = cfg
        self.net = net
        self.extractor = extractor
        self.prompt_bank = prompt_bank
        self.dev = dev
        self._n_devices = 1
        self._start_epoch = 0
        self.prepare_dataloader()
        self.prepare_optimizer()
        self.prepare_scheduler()
        self._init_weights()

    def _init_weights(self):
        """Load or validate checkpoints; avoids accidental overwrite."""
        if self.cfg.resume:
            self.resume()
        elif self.cfg.load_from:
            self.load()
        else:
            if len(os.listdir(self._logdir)) > 0:
                raise RuntimeError(f"Target directory not empty, check to avoid overwrite!\n{self._logdir}")

    @abstractmethod
    def prepare_dataloader(self):
        pass

    def prepare_optimizer(self):
        """Build AdamW optimizer with decoupled weight decay for select params."""
        skip_decay = ['bias', 'Norm', 'temperature']
        param_groups = [
            {'params': [p for n, p in self.net.named_parameters() if not any(nd in n for nd in skip_decay)], 'weight_decay': 1e-2},
            {'params': [p for n, p in self.net.named_parameters() if any(nd in n for nd in skip_decay)], 'weight_decay': 0.0}
        ]
        self.opt = torch.optim.AdamW(param_groups, lr=self.cfg.max_lr)

    def prepare_scheduler(self):
        """Configure LR scheduler based on config choice (linear or cycle)."""
        steps_per_epoch = self.num_batches
        total_steps = self.cfg.num_epochs * steps_per_epoch
        print("steps_per_epoch:", steps_per_epoch)
        print("total_steps:", total_steps)
        if self.cfg.lr_scheduler_type == 'linear':
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.opt,
                total_iters=total_steps,
                last_epoch=-1
            )
        elif self.cfg.lr_scheduler_type == 'cycle':
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.opt,
                max_lr=self.cfg.max_lr,
                total_steps=total_steps,
                final_div_factor=100,
                last_epoch=-1,
                pct_start=2/self.cfg.num_epochs,
            )


    def input(self, voxel, subj_id):
        """Default passthrough input; subclasses may adapt signature."""
        return (voxel, subj_id)

    def train(self):
        """Run the training loop over epochs with periodic evaluation and save."""
        ep = self._start_epoch
        self.losses, self.val_losses, self.lrs = [], [], []
        self._best_sim = 0
        self._best_epoch = 0
        self._val_voxel0 = self._val_image0 = None
        print(f"{self.cfg.model_name} start at epoch {ep} / {self.cfg.num_epochs}")
        bar = tqdm(range(ep, self.cfg.num_epochs))
        for ep in bar:
            self.net.train()
            self.sims_image = 0.
            self.sims_text = 0.
            self.val_sims_image = 0.
            self.val_sims_text = 0.
            self.fwd_percent_correct = 0.
            self.bwd_percent_correct = 0.
            self.val_fwd_percent_correct = 0.
            self.val_bwd_percent_correct = 0.
            self.loss_clip_image_sum = 0.
            self.loss_clip_text_sum = 0.
            self.loss_mse_image_sum = 0.
            self.loss_mse_text_sum = 0.
            self.loss_rec_sum = 0.
            self.loss_cyc_sum = 0.
            self.val_loss_clip_image_sum = 0.
            self.val_loss_clip_text_sum = 0.
            self.val_loss_mse_image_sum = 0.
            self.val_loss_mse_text_sum = 0.
            self.val_loss_rec_sum = 0.
            self.val_loss_cyc_sum = 0.
            self.train_epoch(ep)
            self.log_train()
            if ep % self.cfg.eval_interval == 0:
                self.eval_epoch(ep)
                self.log_val()
            bar.set_postfix({"epoch": ep, "lr": self.logs["train/lr"], "loss": self.logs["train/loss"]})
            if ep > 200:
                if ep % self.cfg.ckpt_interval == 0 or ep == self.cfg.num_epochs-1:
                    self.save(ep)

    @abstractmethod
    def train_epoch(self, epoch):
        pass

    def train_step(self, voxel, image, captions, subj_id):
        """One gradient update step for a single subject minibatch."""
        loss = 0.
        self.opt.zero_grad()

        if self.cfg.use_image_aug:
            image = data.img_augment(image)
        clip_image = self.extractor.embed_image(image).float()   
        clip_text = self.extractor.embed_text(captions).float()
        results = self.net(self.input(voxel, subj_id), subj_id)

        # image clip loss
        clip_image_pred = results[0]
        clip_image_pred_nonorm = clip_image_pred.flatten(1)
        clip_image_pred_norm = nn.functional.normalize(clip_image_pred_nonorm, dim=-1)
        clip_image_nonorm = clip_image.flatten(1)
        clip_image_norm = nn.functional.normalize(clip_image_nonorm, dim=-1)
        loss_clip_image = utils.soft_clip_loss(
            clip_image_pred_norm,
            clip_image_norm,
        )
        utils.check_loss(loss_clip_image, "loss_clip_image")
        loss += loss_clip_image
        self.loss_clip_image_sum += loss_clip_image.item()


        if self.cfg.mse_mult:
            loss_mse_image = nn.MSELoss()(clip_image_pred_norm, clip_image_norm)
            utils.check_loss(loss_mse_image, "loss_mse_image")
            loss += self.cfg.mse_mult * loss_mse_image
            self.loss_mse_image_sum += loss_mse_image.item()

        # text clip loss
        clip_text_pred = results[1]
        clip_text_pred_nonorm = clip_text_pred.flatten(1)
        clip_text_pred_norm = nn.functional.normalize(clip_text_pred_nonorm, dim=-1)
        clip_text_nonorm = clip_text.flatten(1)
        clip_text_norm = nn.functional.normalize(clip_text_nonorm, dim=-1)
        loss_clip_text = utils.soft_clip_loss(
            clip_text_pred_norm,
            clip_text_norm,
        )
        utils.check_loss(loss_clip_text, "loss_clip_text")
        loss += loss_clip_text
        self.loss_clip_text_sum += loss_clip_text.item()

        # text mse loss
        if self.cfg.mse_mult:
            loss_mse_text = nn.MSELoss()(clip_text_pred_norm, clip_text_norm)
            utils.check_loss(loss_mse_text, "loss_mse_text")
            loss += self.cfg.mse_mult * loss_mse_text
            self.loss_mse_text_sum += loss_mse_text.item()

        # brain reconstruction loss
        if self.cfg.rec_mult:
            voxel_rec = results[2]
            loss_rec = nn.MSELoss()(voxel, voxel_rec)
            utils.check_loss(loss_rec, "loss_rec")
            loss += self.cfg.rec_mult * loss_rec            
            self.loss_rec_sum += loss_rec.item()
        
        # cycle loss
        if self.cfg.cyc_mult:
            loss_cyc = results[3]
            utils.check_loss(loss_cyc, "loss_cyc")
            loss += self.cfg.cyc_mult * loss_cyc
            self.loss_cyc_sum += loss_cyc.item()
        
        utils.check_loss(loss)
        loss.backward()
        self.opt.step()

        self.losses.append(loss.item())
        self.lrs.append(self.opt.param_groups[0]['lr'])
        self.lr_scheduler.step()

        self.sims_image += nn.functional.cosine_similarity(clip_image_norm,clip_image_pred_norm).mean().item()
        self.sims_text += nn.functional.cosine_similarity(clip_text_norm,clip_text_pred_norm).mean().item()

        # forward and backward top 1 accuracy
        labels = torch.arange(len(clip_image_norm)).to(self.dev) 
        self.fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_image_pred_norm, clip_image_norm), labels, k=1)
        self.bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_image_norm, clip_image_pred_norm), labels, k=1)


    def train_multi_step(self, voxels, images, captionss):
        """Multi-subject step that accumulates losses before one optimizer step."""
        subject_ids = [1, 2, 5, 7]
        loss = 0.
        self.opt.zero_grad()


        images_aug = [data.img_augment(img) if self.cfg.use_image_aug else img for img in images]

        clip_images = [self.extractor.embed_image(img).float() for img in images_aug]
        clip_texts = [self.extractor.embed_text(caps).float() for caps in captionss]

        results = [self.net(self.input(vox.to(self.dev), sid), f'subj{sid}') for vox, sid in zip(voxels, subject_ids)]

        sims_image = []
        sims_text = []
        fwd_acc = []
        bwd_acc = []

        for i in range(4):
            clip_image_pred = results[i][0]
            clip_image_pred_nonorm = clip_image_pred.flatten(1)
            clip_image_pred_norm = nn.functional.normalize(clip_image_pred_nonorm, dim=-1)
            clip_image_nonorm = clip_images[i].flatten(1)
            clip_image_norm = nn.functional.normalize(clip_image_nonorm, dim=-1)
            loss_clip_image = utils.soft_clip_loss(
                clip_image_pred_norm,
                clip_image_norm,
            )
            utils.check_loss(loss_clip_image, "loss_clip_image")
            loss += loss_clip_image
            self.loss_clip_image_sum += loss_clip_image.item()
            if self.cfg.mse_mult:
                loss_mse_image = nn.MSELoss()(clip_image_pred_norm, clip_image_norm)
                utils.check_loss(loss_mse_image, "loss_mse_image")
                loss += self.cfg.mse_mult * loss_mse_image
                self.loss_mse_image_sum += loss_mse_image.item()

            # text
            clip_text_pred = results[i][1]
            clip_text_pred_nonorm = clip_text_pred.flatten(1)
            clip_text_pred_norm = nn.functional.normalize(clip_text_pred_nonorm, dim=-1)
            clip_text_nonorm = clip_texts[i].flatten(1)
            clip_text_norm = nn.functional.normalize(clip_text_nonorm, dim=-1)
            loss_clip_text = utils.soft_clip_loss(
                clip_text_pred_norm,
                clip_text_norm,
            )
            utils.check_loss(loss_clip_text, "loss_clip_text")
            loss += loss_clip_text
            self.loss_clip_text_sum += loss_clip_text.item()
            if self.cfg.mse_mult:
                loss_mse_text = nn.MSELoss()(clip_text_pred_norm, clip_text_norm)
                utils.check_loss(loss_mse_text, "loss_mse_text")
                loss += self.cfg.mse_mult * loss_mse_text
                self.loss_mse_text_sum += loss_mse_text.item()

            # sim
            sims_image.append(nn.functional.cosine_similarity(clip_image_norm, clip_image_pred_norm).mean().item())
            sims_text.append(nn.functional.cosine_similarity(clip_text_norm, clip_text_pred_norm).mean().item())

            # accuracy
            labels = torch.arange(len(clip_image_norm)).to(self.dev)
            fwd_acc.append(utils.topk(utils.batchwise_cosine_similarity(clip_image_pred_norm, clip_image_norm), labels, k=1))
            bwd_acc.append(utils.topk(utils.batchwise_cosine_similarity(clip_image_norm, clip_image_pred_norm), labels, k=1))

        self.sims_image += sum(sims_image) / 4
        self.sims_text += sum(sims_text) / 4
        self.fwd_percent_correct += sum(fwd_acc) / 4
        self.bwd_percent_correct += sum(bwd_acc) / 4

        utils.check_loss(loss)
        loss.backward()
        self.opt.step()

        self.losses.append(loss.item())
        self.lrs.append(self.opt.param_groups[0]['lr'])
        self.lr_scheduler.step()

    @abstractmethod
    def eval_epoch(self, epoch):
        pass

    def eval_step(self, voxel, image, captions, subj_id):
        """Validation step mirroring train_step without gradients."""
        val_loss = 0.
        with torch.no_grad():
            # used for reconstruction
            if self._val_image0 is None:
                self._val_image0 = image.detach().clone()
                self._val_voxel0 = voxel.detach().clone()

            clip_image = self.extractor.embed_image(image).float()
            clip_text = self.extractor.embed_text(captions).float()

            results = self.net(self.input(voxel.to(self.dev), subj_id).float(), 'subj{}'.format(subj_id.item()))

            # image clip loss
            clip_image_pred = results[0]
            clip_image_pred_nonorm = clip_image_pred.flatten(1)
            clip_image_pred_norm = nn.functional.normalize(clip_image_pred_nonorm, dim=-1)
            clip_image_nonorm = clip_image.flatten(1)
            clip_image_norm = nn.functional.normalize(clip_image_nonorm, dim=-1)
            val_loss_clip_image = utils.soft_clip_loss(
                clip_image_pred_norm,
                clip_image_norm,
            )
            val_loss += val_loss_clip_image
            self.val_loss_clip_image_sum += val_loss_clip_image.item()

            # image mse loss
            if self.cfg.mse_mult:
                val_loss_mse_image = nn.MSELoss()(clip_image_pred_norm, clip_image_norm)
                val_loss += self.cfg.mse_mult * val_loss_mse_image
                self.val_loss_mse_image_sum += val_loss_mse_image.item()

            # text clip loss
            clip_text_pred = results[1]
            clip_text_pred_nonorm = clip_text_pred.flatten(1)
            clip_text_pred_norm = nn.functional.normalize(clip_text_pred_nonorm, dim=-1)
            clip_text_nonorm = clip_text.flatten(1)
            clip_text_norm = nn.functional.normalize(clip_text_nonorm, dim=-1)
            val_loss_clip_text = utils.soft_clip_loss(
                clip_text_pred_norm,
                clip_text_norm,
            )
            val_loss += val_loss_clip_text
            self.val_loss_clip_text_sum += val_loss_clip_text.item()

            # text mse loss
            if self.cfg.mse_mult:
                val_loss_mse_text = nn.MSELoss()(clip_text_pred_norm, clip_text_norm)
                val_loss += self.cfg.mse_mult * val_loss_mse_text
                self.val_loss_mse_text_sum += val_loss_mse_text.item()

            # brain reconstruction loss
            if self.cfg.rec_mult:
                voxel_rec = results[2]
                val_loss_rec = nn.MSELoss()(voxel, voxel_rec)
                val_loss += self.cfg.rec_mult * val_loss_rec
                self.val_loss_rec_sum += val_loss_rec.item()

            # cycle loss
            if self.cfg.cyc_mult:
                loss_cyc = results[3]
                val_loss_cyc = loss_cyc
                val_loss += self.cfg.cyc_mult * val_loss_cyc
                self.val_loss_cyc_sum += val_loss_cyc.item()

            utils.check_loss(val_loss)
            self.val_losses.append(val_loss.item())

            self.val_sims_image += nn.functional.cosine_similarity(clip_image_norm,clip_image_pred_norm).mean().item()
            self.val_sims_text += nn.functional.cosine_similarity(clip_text_norm,clip_text_pred_norm).mean().item()
            
            labels = torch.arange(len(clip_image_norm)).to(self.dev) 
            self.val_fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_image_pred_norm, clip_image_norm), labels, k=1)
            self.val_bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_image_norm, clip_image_pred_norm), labels, k=1)

    def vis(self,):
        pass

    def save_ckpt(self, tag, epoch):
        """Save a lightweight checkpoint with epoch and model state."""
        ckpt_path = self._logdir+f'/{tag}.pth'
        print(f'--- saving model: {ckpt_path} ---',flush=True)

        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.net.state_dict(),
                }, ckpt_path)
        except:
            print("Couldn't save... moving on to prevent crashing.")

    def save(self, epoch):
        """Save last and, if improved, best checkpoints based on similarity."""
        self.save_ckpt(f'last', epoch)
        # save best model
        current_sim = (self.val_sims_image + self.val_sims_text) / (self.val_i + 1) if hasattr(self, 'val_i') else 0
        if current_sim > self._best_sim:
            self._best_sim = current_sim
            self._best_epoch = epoch
            self.save_ckpt(f'best', epoch)
        else:
            print(f'Not best - current_similarity: {current_sim:.3f} @ epoch {epoch}, best_similarity: {self._best_sim:.3f} @ epoch {self._best_epoch}')
                
    def load(self,):
        """Load model weights from a provided checkpoint path."""
        print("\n--- load from ckpt: {} ---\n".format(self.cfg.load_from))
        checkpoint = torch.load(self.cfg.load_from, map_location='cpu')
        self.net.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("loaded keys", checkpoint['model_state_dict'].keys())
        del checkpoint

    def resume(self,):
        """Resume training from the last checkpoint in the log directory."""
        state_path = os.path.join(self._logdir, "last")
        print(f"\n--- resuming from {state_path} ---\n")
        self.net.load_state_dict(state_path)

        ckpt_path = self._logdir+'/last.pth'
        print(f"\n--- Read resume epoch from {ckpt_path} ---\n")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        self._start_epoch = checkpoint['epoch']
        print(">>> Resume at Epoch", self._start_epoch)
        del checkpoint
    
    def log_train(self):
        """Aggregate and store training metrics for the current epoch."""
        self.logs = {
            "train/loss": np.mean(self.losses[-(self.train_i+1):]),
            "train/lr": self.lrs[-1],
            "train/num_steps": len(self.losses),
            "train/cosine_sim_image": self.sims_image / (self.train_i + 1),
            "train/cosine_sim_text": self.sims_text / (self.train_i + 1),
            "train/fwd_pct_correct": self.fwd_percent_correct / (self.train_i + 1),
            "train/bwd_pct_correct": self.bwd_percent_correct / (self.train_i + 1),
            "train/loss_clip_image": self.loss_clip_image_sum / (self.train_i + 1),
            "train/loss_clip_text": self.loss_clip_text_sum / (self.train_i + 1),
            "train/loss_mse_image": self.loss_mse_image_sum / (self.train_i + 1),
            "train/loss_mse_text": self.loss_mse_text_sum / (self.train_i + 1),
            "train/loss_rec": self.loss_rec_sum / (self.train_i + 1),
            "train/loss_cyc": self.loss_cyc_sum / (self.train_i + 1),
        }

    def log_val(self):
        """Aggregate and store validation metrics for the current evaluation."""
        self.logs.update({
            "val/loss": np.mean(self.val_losses[-(self.val_i+1):]),
            "val/num_steps": len(self.val_losses),
            "val/cosine_sim_image": self.val_sims_image / (self.val_i + 1),
            "val/cosine_sim_text": self.val_sims_text / (self.val_i + 1),
            "val/val_fwd_pct_correct": self.val_fwd_percent_correct / (self.val_i + 1),
            "val/val_bwd_pct_correct": self.val_bwd_percent_correct / (self.val_i + 1),
            "val/loss_clip_image": self.val_loss_clip_image_sum / (self.val_i + 1),
            "val/loss_clip_text": self.val_loss_clip_text_sum / (self.val_i + 1),
            "val/loss_mse_image": self.val_loss_mse_image_sum / (self.val_i + 1),
            "val/loss_mse_text": self.val_loss_mse_text_sum / (self.val_i + 1),
            "val/loss_rec": self.val_loss_rec_sum / (self.val_i + 1),
            "val/loss_cyc": self.val_loss_cyc_sum / (self.val_i + 1),
        })


class Trainer_Multi(CoreHandler):
    """Multi-subject trainer that zips per-subject dataloaders for joint steps."""
    def __init__(self, args, voxel2clip, clip_extractor, prompts_list, device) -> None:
        super().__init__(args, voxel2clip, clip_extractor, prompts_list, device)

    def prepare_dataloader(self):
        """Construct train/val dataloaders for each subject in config."""
        # Prepare data and dataloader
        print("Preparing data and dataloader...")
        self.train_dls = [] # tarin_dls contains all subjects separately
        self.val_dls = [] # tarin_dls contains all subjects separately

        for subj in self.cfg.subj_list:
            train_dl, val_dl = data.get_dls(
                subject=subj,
                data_path=self.cfg.data_path,
                batch_size=self.cfg.batch_size,
                val_batch_size=self.cfg.val_batch_size,
                num_workers=self.cfg.num_workers,
                pool_type=self.cfg.pool_type,
                pool_num=self.cfg.pool_num,
                length=self.cfg.length,
                seed=self.cfg.seed,
            )
            self.train_dls.append(train_dl)
            self.val_dls.append(val_dl)
        
        self.num_batches = len(self.train_dls[0])

    def input(self, voxel, subj_id):
        # adapting need to know subj_id
        return voxel

    def train_epoch(self, epoch):
        """One epoch over zipped subject batches with shared step."""
        # train loop
        for train_i, datas in enumerate(zip(*self.train_dls)):
            self.train_i = train_i
            repeat_index = train_i % 3
            subject_ids = [1, 2, 5, 7]
            subject_data = []
            for data, sid in zip(datas, subject_ids):
                voxel, image, coco, subj_id = data
                voxel = voxel[:, repeat_index, ...].float()
                subj_id = subj_id[[0], ...]
                coco_ids = coco.squeeze().tolist()
                current_prompts_list = [self.prompt_bank[coco_id] for coco_id in coco_ids]
                captions = [prompts[repeat_index]['caption'] for prompts in current_prompts_list]
                subject_data.append((voxel, image, captions))
            voxels, images, captionss = zip(*subject_data)
            self.train_multi_step(list(voxels), list(images), list(captionss))


    def eval_epoch(self, epoch):
        """Run evaluation for all subject dataloaders."""
        print("Evaluating...")
        self.net.eval()
        for val_dl in self.val_dls:
            for val_i, data_i in enumerate(val_dl):
                self.val_i = val_i
                repeat_index = val_i % 3  # randomly choose the one in the repeated three
                voxel, image, coco, subj_id = data_i
                voxel = torch.mean(voxel, axis=1)
                subj_id = subj_id[[0], ...]

                coco_ids = coco.squeeze().tolist()
                current_prompts_list = [self.prompt_bank[coco_id] for coco_id in coco_ids]
                captions = [prompts[repeat_index]['caption'] for prompts in current_prompts_list]

                print(">>>Subject:{} Epoch{} | Eval{} | voxel: {}".format(subj_id, epoch, val_i, voxel.shape), flush=True)
                self.eval_step(voxel, image, captions, subj_id)


