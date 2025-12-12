from dataclasses import dataclass
from typing import Dict, List
import torch
import mlflow
from .augmentation import augment_batch


@dataclass
class TrainConfig:
    num_epochs = 40
    batch_size = 4
    num_workers = 4
    lr = 0.005
    weight_decay = 1e-4
    print_every = 50
    momentum = 0.9
    tracking_uri = "file:///media/sdb1/mlflow"

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, model, train_conf: TrainConfig):
        self.model = model
        self.conf = train_conf
        self.device = train_conf.device
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(params,lr=train_conf.lr,momentum=train_conf.momentum,weight_decay=train_conf.weight_decay,)

    def _to_device(self, images, targets):
        images = [img.to(self.device) for img in images]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        return images, targets

    def _forward_optional_masks(self, images, targets):
        if targets is not None and hasattr(self.model, "roi_heads") and self.model.roi_heads.has_mask():
            if not all(("masks" in t) for t in targets):
                rh = self.model.roi_heads
                saved = (rh.mask_roi_pool, rh.mask_head, rh.mask_predictor)
                rh.mask_roi_pool, rh.mask_head, rh.mask_predictor = None, None, None
                try:
                    return self.model(images, targets)
                finally:
                    rh.mask_roi_pool, rh.mask_head, rh.mask_predictor = saved
        return self.model(images, targets)

    def train_one_epoch(self, loader, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0

        for step, (images, targets) in enumerate(loader, 1):
            images, targets = augment_batch(images, targets)
            images, targets = self._to_device(images, targets)

            #loss_dict = self.model(images, targets)
            loss_dict = self._forward_optional_masks(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            if step % self.conf.print_every == 0 or step == len(loader):
                print(f"[epoch {epoch + 1:03d}/{self.conf.num_epochs:03d}] "f"step {step}/{len(loader)} loss {loss.item():.4f}")

        return running_loss / len(loader)

    def eval_one_epoch(self, loader) -> float:
        self.model.train()
        running_loss = 0.0

        with torch.no_grad():
            for images, targets in loader:
                images, targets = self._to_device(images, targets)
                #loss_dict = self.model(images, targets)
                loss_dict = self._forward_optional_masks(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                running_loss += loss.item()

        return running_loss / len(loader)

    def run(self,train_loader,val_loader,experiment_name: str | None = None,) -> Dict[str, List[float]]:
        history = {"train_loss": [], "val_loss": []}

        if experiment_name is not None:
            mlflow.set_tracking_uri(self.conf.tracking_uri)
            mlflow.set_experiment(experiment_name)
            run_ctx = mlflow.start_run()
        else:
            run_ctx = None

        if run_ctx is not None:
            run_ctx.__enter__()
            mlflow.log_params(
                {
                    "batch_size": self.conf.batch_size,
                    "lr": self.conf.lr,
                    "weight_decay": self.conf.weight_decay,
                    "num_epochs": self.conf.num_epochs,
                }
            )

        for epoch in range(self.conf.num_epochs):
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_loss = self.eval_one_epoch(val_loader)

            print(
                f"epoch {epoch + 1:03d}/{self.conf.num_epochs:03d}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}"
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if run_ctx is not None:
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)

        if run_ctx is not None:
            mlflow.pytorch.log_model(self.model, "model")
            run_ctx.__exit__(None, None, None)

        return history
