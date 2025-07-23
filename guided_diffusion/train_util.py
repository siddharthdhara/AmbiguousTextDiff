import copy
import functools
import os
import blobfile as bf
import time
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from contextlib import nullcontext

from . import dist_util, logger
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

INITIAL_LOG_LOSS_SCALE = 20.0

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        prior,
        posterior,
        classifier,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        text_encoder=None,
        grad_clip=-1.,
        disable_text_guidance=False,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.prior = prior
        self.posterior = posterior
        self.classifier = classifier
        self.text_encoder = text_encoder
        self.disable_text_guidance = disable_text_guidance
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion, diffusion.num_timesteps)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.grad_clip = grad_clip

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        params_to_optimize = list(self.model.parameters())
        if self.text_encoder and any(p.requires_grad for p in self.text_encoder.parameters()):
            params_to_optimize.extend(list(p for p in self.text_encoder.parameters() if p.requires_grad))
        if self.prior and any(p.requires_grad for p in self.prior.parameters()):
            params_to_optimize.extend(list(p for p in self.prior.parameters() if p.requires_grad))
        if self.posterior and any(p.requires_grad for p in self.posterior.parameters()):
            params_to_optimize.extend(list(p for p in self.posterior.parameters() if p.requires_grad))

        self.opt = AdamW(params_to_optimize, lr=self.lr, weight_decay=self.weight_decay)
        
        # === THE FIX IS HERE ===
        # Initialize as None. We will create it later.
        self.ema_params = None
        
        self._load_and_sync_parameters()

        if dist.get_world_size() > 1:
            self.use_ddp = True
            # ... DDP setup
        else:
            self.use_ddp = False
            self.ddp_model = self.model
            self.ddp_text_encoder = self.text_encoder
            self.ddp_prior = self.prior
            self.ddp_posterior = self.posterior
            self.ddp_classifier = self.classifier

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if not resume_checkpoint: return
        self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
        
        if dist.get_rank() == 0:
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            state_dict = dist_util.load_state_dict(resume_checkpoint, map_location=dist_util.dev())
            self.model.load_state_dict(state_dict)
            self._load_ema_parameters()
            self._load_optimizer_state()

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if not main_checkpoint: return
        opt_checkpoint = bf.join(bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt")
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
            self.opt.load_state_dict(state_dict)

    def _load_ema_parameters(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if not main_checkpoint: return
        for rate in self.ema_rate:
            ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
            if ema_checkpoint:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(ema_checkpoint, map_location=dist_util.dev())
                self.ema_params = [state_dict[name].clone().detach() for name, _ in self.model.named_parameters()]

    def run_loop(self):
        data_iter = iter(self.data)
        while not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps:
            try:
                data_item = next(data_iter)
            except StopIteration:
                data_iter = iter(self.data)
                data_item = next(data_iter)

            batch_images = data_item["image"].to(dist_util.dev())
            batch_masks = data_item["mask"].to(dist_util.dev())
            batch_combined = th.cat((batch_images, batch_masks), dim=1)

            cond = {"text": data_item["text"]}
            if "y" in data_item:
                cond["y"] = data_item["y"].to(dist_util.dev())

            self.run_step(batch_combined, cond)

            if self.step > 0 and self.step % self.log_interval == 0:
                logger.dumpkvs()

            if self.step > 0 and self.step % self.save_interval == 0:
                self.save()

            self.step += 1

        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        
        if self.grad_clip > 0:
            all_params = []
            for param_group in self.opt.param_groups:
                all_params.extend(param_group['params'])
            th.nn.utils.clip_grad_norm_(all_params, self.grad_clip)
        
        self.opt.step()
        self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.opt.zero_grad()
        for i in range(0, self.batch_size, self.microbatch):
            micro_combined = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {"text": cond["text"][i : i + self.microbatch]}
            if "y" in cond:
                micro_cond["y"] = cond["y"][i : i + self.microbatch].to(dist_util.dev())

            if self.text_encoder:
                if self.disable_text_guidance:
                # For ablation, use a zero tensor as embedding
                    text_emb_shape = (micro_cond["text"].__len__(), self.text_encoder.output_dim)
                    micro_cond["text_embeddings"] = th.zeros(text_emb_shape, device=dist_util.dev())
                else:
                # Normal operation
                    micro_cond["text_embeddings"] = self.ddp_text_encoder(micro_cond["text"])
            
            
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro_combined.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses_segmentation,
                self.ddp_model, self.ddp_classifier, self.ddp_prior, self.ddp_posterior,
                micro_combined, t, model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses1 = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses1 = compute_losses()
            
            losses = losses1[0]
            loss = (losses["loss"] * weights).mean()
            log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})
            loss.backward()

    def _update_ema(self):
        # === THE FIX IS HERE ===
        # If this is the first time, create the ema_params list now that the model is stable.
        if self.ema_params is None:
            self.ema_params = [
                copy.deepcopy(p.data) for p in self.model.parameters()
            ]

        for rate in self.ema_rate:
            for ema_p, p in zip(self.ema_params, self.model.parameters()):
                ema_p.mul_(rate).add_(p.data, alpha=1 - rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps: return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.grad_clip > 0:
            all_params = []
            for param_group in self.opt.param_groups:
                all_params.extend(param_group['params'])
            params_with_grad = [p for p in all_params if p.grad is not None]
            if params_with_grad:
                grad_norm = th.norm(th.stack([th.norm(p.grad.detach(), 2.0) for p in params_with_grad]), 2.0)
                logger.logkv_mean("grad_norm", grad_norm.item())

    def save(self):
        if dist.get_rank() != 0: return
        step = self.step + self.resume_step
        logger.log("saving main model...")
        with bf.BlobFile(bf.join(get_blob_logdir(), f"model{step:06d}.pt"), "wb") as f:
            th.save(self.model.state_dict(), f)
        for rate in self.ema_rate:
            logger.log(f"saving model ema {rate}...")
            ema_state_dict = {}
            for i, (name, _p) in enumerate(self.model.named_parameters()):
                ema_state_dict[name] = self.ema_params[i]
            with bf.BlobFile(bf.join(get_blob_logdir(), f"ema_{rate}_{step:06d}.pt"), "wb") as f:
                th.save(ema_state_dict, f)
        if self.prior:
            with bf.BlobFile(bf.join(get_blob_logdir(), f"prior{step:06d}.pt"), "wb") as f:
                th.save(self.prior.state_dict(), f)
        if self.posterior:
            with bf.BlobFile(bf.join(get_blob_logdir(), f"posterior{step:06d}.pt"), "wb") as f:
                th.save(self.posterior.state_dict(), f)
        if self.text_encoder and any(p.requires_grad for p in self.text_encoder.parameters()):
            with bf.BlobFile(bf.join(get_blob_logdir(), f"text_encoder{step:06d}.pt"), "wb") as f:
                th.save(self.text_encoder.state_dict(), f)
        with bf.BlobFile(bf.join(get_blob_logdir(), f"opt{step:06d}.pt"), "wb") as f:
            th.save(self.opt.state_dict(), f)

def parse_resume_step_from_filename(filename):
    split = filename.split("model")
    if len(split) < 2: return 0
    split1 = split[-1].split(".")[0]
    try: return int(split1)
    except ValueError: return 0

def get_blob_logdir():
    return logger.get_dir()

def find_resume_checkpoint(): return None
def find_ema_checkpoint(main_checkpoint, step, rate): return None

def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        if th.isnan(values).any():
            logger.log(f"WARNING: NaN detected in loss '{key}'")
            continue
        logger.logkv_mean(key, values.mean().item())