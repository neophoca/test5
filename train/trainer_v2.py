"""
trainer for Mask R-CNN-style detectors

- MLflow logging (config.tracking_uri)
- AMP (autocast + GradScaler)
- LR warmup + scheduler (none|cosine|step)
- EMA weights for evaluation
- Gradient clipping
- BatchNorm freezing 

    conf = TrainConfig.from_json("./train.json")
    trainer = Trainer(model, conf)
    history = trainer.run(train_loader, val_loader, experiment_name="karyo", run_name="attfpn")
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import copy
import json
import math
import torch
import mlflow
import mlflow.pytorch
from .augmentation_v2 import augment_batch


@dataclass
class TrainConfig:
    num_epochs: int = 40
    batch_size: int = 4
    num_workers: int = 4

    lr: float = 0.005
    weight_decay: float = 1e-4
    momentum: float = 0.9

    print_every: int = 50

    tracking_uri: str = ""

    amp: bool = True
    grad_clip: float = 1.0

    ema_decay: float = 0.0

    warmup_iters: int = 0
    scheduler: str = "cosine"  # none|cosine|step
    min_lr: float = 1e-6
    step_milestones: Tuple[int, ...] = (27, 33)
    step_gamma: float = 0.1

    freeze_bn: bool = True

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def from_json(path: str) -> "TrainConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "step_milestones" in data and isinstance(data["step_milestones"], list):
            data["step_milestones"] = tuple(int(x) for x in data["step_milestones"])
        return TrainConfig(**data)


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = float(decay)
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        d = self.decay
        msd = model.state_dict()
        esd = self.ema.state_dict()
        for k, v in esd.items():
            mv = msd[k]
            if not torch.is_floating_point(v):
                esd[k] = mv
            else:
                esd[k].mul_(d).add_(mv, alpha=1.0 - d)
        self.ema.load_state_dict(esd, strict=True)

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, sd):
        self.ema.load_state_dict(sd, strict=True)


def _freeze_bn_(model: torch.nn.Module):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()


class Trainer:
    def __init__(self, model: torch.nn.Module, train_conf: TrainConfig):
        self.model = model
        self.conf = train_conf
        self.device = train_conf.device

        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            params,
            lr=train_conf.lr,
            momentum=train_conf.momentum,
            weight_decay=train_conf.weight_decay,
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=bool(train_conf.amp and self.device.type == "cuda"))

        self.ema: Optional[ModelEMA] = None
        if train_conf.ema_decay and train_conf.ema_decay > 0.0:
            self.ema = ModelEMA(self.model, decay=train_conf.ema_decay)

        self._sched_mode = (train_conf.scheduler or "none").lower()
        self._sched_epoch = None
        self._sched_iter = None
        self._global_step = 0
        self._iters_total = None

    def _to_device(self, images, targets):
        images = [img.to(self.device) for img in images]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        return images, targets

    def _forward_optional_masks(self, images, targets):
        if targets is not None and hasattr(self.model, "roi_heads") and self.model.roi_heads.has_mask():
            has_masks = all(("masks" in t) for t in targets)
            if not has_masks:
                rh = self.model.roi_heads
                saved = (rh.mask_roi_pool, rh.mask_head, rh.mask_predictor)
                rh.mask_roi_pool, rh.mask_head, rh.mask_predictor = None, None, None
                try:
                    return self.model(images, targets)
                finally:
                    rh.mask_roi_pool, rh.mask_head, rh.mask_predictor = saved
        return self.model(images, targets)

    def _init_schedulers(self, steps_per_epoch: int):
        iters_total = int(self.conf.num_epochs) * int(steps_per_epoch)
        self._iters_total = iters_total

        if self._sched_mode == "step":
            self._sched_epoch = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=list(self.conf.step_milestones),
                gamma=float(self.conf.step_gamma),
            )
            self._sched_iter = None
            return

        if self._sched_mode == "cosine":
            tmax = max(1, iters_total - int(self.conf.warmup_iters))
            self._sched_iter = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=tmax,
                eta_min=float(self.conf.min_lr),
            )
            self._sched_epoch = None
            return

        self._sched_epoch = None
        self._sched_iter = None

    def _apply_warmup(self):
        w = int(self.conf.warmup_iters)
        if w <= 0:
            return
        s = self._global_step
        if s <= w:
            scale = max(1e-8, float(s) / float(w))
            for pg in self.optimizer.param_groups:
                pg["lr"] = float(self.conf.lr) * scale

    def _sched_step(self):
        if self._sched_mode == "cosine" and self._sched_iter is not None:
            if self._global_step > int(self.conf.warmup_iters):
                self._sched_iter.step()

    def _sched_epoch_step(self):
        if self._sched_mode == "step" and self._sched_epoch is not None:
            self._sched_epoch.step()

    def train_one_epoch(self, loader, epoch: int) -> float:
        self.model.train()
        if self.conf.freeze_bn:
            _freeze_bn_(self.model)

        running_loss = 0.0
        n = 0

        for step, (images, targets) in enumerate(loader, 1):
            images, targets = augment_batch(images, targets)
            images, targets = self._to_device(images, targets)

            self._global_step += 1
            self._apply_warmup()

            with torch.autocast(device_type=self.device.type, enabled=bool(self.conf.amp and self.device.type == "cuda"), dtype=torch.float16):
                loss_dict = self._forward_optional_masks(images, targets)
                loss = sum(v for v in loss_dict.values())

            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                if self.conf.grad_clip and self.conf.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.conf.grad_clip))
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.conf.grad_clip and self.conf.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.conf.grad_clip))
                self.optimizer.step()

            if self.ema is not None:
                self.ema.update(self.model)

            self._sched_step()

            running_loss += float(loss.item())
            n += 1

            if step % int(self.conf.print_every) == 0 or step == len(loader):
                print(f"[epoch {epoch + 1:03d}/{int(self.conf.num_epochs):03d}] step {step}/{len(loader)} loss {loss.item():.4f}")

        return running_loss / max(1, n)

    def eval_one_epoch(self, loader) -> float:
        self.model.train()
        if self.conf.freeze_bn:
            _freeze_bn_(self.model)

        running_loss = 0.0
        n = 0

        with torch.no_grad():
            for images, targets in loader:
                images, targets = self._to_device(images, targets)
                with torch.autocast(device_type=self.device.type, enabled=bool(self.conf.amp and self.device.type == "cuda"), dtype=torch.float16):
                    loss_dict = self._forward_optional_masks(images, targets)
                    loss = sum(v for v in loss_dict.values())
                running_loss += float(loss.item())
                n += 1

        return running_loss / max(1, n)

    def _eval_with_ema(self, loader) -> float:
        if self.ema is None:
            return self.eval_one_epoch(loader)
        sd = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.ema.state_dict(), strict=True)
        try:
            return self.eval_one_epoch(loader)
        finally:
            self.model.load_state_dict(sd, strict=True)

    def run(self, train_loader, val_loader, experiment_name: str | None = None, steps_per_epoch: int | None = None) -> Dict[str, List[float]]:
        history = {"train_loss": [], "val_loss": [], "lr": []}

        self.model.to(self.device)
        if self.conf.freeze_bn:
            _freeze_bn_(self.model)

        self._init_schedulers(steps_per_epoch=int(steps_per_epoch) if steps_per_epoch is not None else len(train_loader))

        run_ctx = None
        if experiment_name is not None:
            mlflow.set_tracking_uri(self.conf.tracking_uri)
            mlflow.set_experiment(experiment_name)
            run_ctx = mlflow.start_run()
            mlflow.log_params({
                "num_epochs": int(self.conf.num_epochs),
                "batch_size": int(self.conf.batch_size),
                "lr": float(self.conf.lr),
                "weight_decay": float(self.conf.weight_decay),
                "momentum": float(self.conf.momentum),
                "amp": bool(self.conf.amp),
                "grad_clip": float(self.conf.grad_clip),
                "ema_decay": float(self.conf.ema_decay),
                "warmup_iters": int(self.conf.warmup_iters),
                "scheduler": self._sched_mode,
                "min_lr": float(self.conf.min_lr),
                "step_milestones": list(self.conf.step_milestones),
                "step_gamma": float(self.conf.step_gamma),
                "freeze_bn": bool(self.conf.freeze_bn),
            })

        for epoch in range(int(self.conf.num_epochs)):
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_loss = self._eval_with_ema(val_loader)

            lr_now = float(self.optimizer.param_groups[0]["lr"])
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["lr"].append(lr_now)

            if run_ctx is not None:
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("lr", lr_now, step=epoch)

            print(f"[epoch {epoch + 1:03d}/{int(self.conf.num_epochs):03d}] train={train_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.6g}")

            self._sched_epoch_step()

        if run_ctx is not None:
            mlflow.pytorch.log_model(self.model, "model")
            run_ctx.__exit__(None, None, None)

        return history
