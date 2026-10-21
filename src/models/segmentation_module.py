from collections.abc import Callable

import segmentation_models_pytorch as smp
import torch
from torchmetrics import MaxMetric, MeanMetric, MetricCollection
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex

from src.models.base_module import BaseLitModule


class SegmentationLitModule(BaseLitModule):
    """`LightningModule` for multiclass semantic segmentation.

    The network (e.g. any `segmentation_models_pytorch` model), optimizer and scheduler are
    defined in the config, see `configs/model/segmentation.yaml`.

    Loss is `cross-entropy + dice_weight * dice`. Logged metrics are mean IoU (`miou`) and mean
    Dice (`dice`) over classes, ignoring pixels equal to `ignore_index`.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: Callable[..., torch.optim.Optimizer],
        scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler] | None,
        num_classes: int,
        ignore_index: int = 255,
        dice_weight: float = 1.0,
        compile: bool = False,
    ) -> None:
        """Initialize a `SegmentationLitModule`.

        :param net: The model to train, returning logits of shape (B, num_classes, H, W).
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param num_classes: The number of classes.
        :param ignore_index: Mask value ignored by the loss and metrics.
        :param dice_weight: Weight of the dice loss term (0 disables it).
        :param compile: Whether to `torch.compile` the model.
        """
        super().__init__(net=net, optimizer=optimizer, scheduler=scheduler, compile=compile)

        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_loss = smp.losses.DiceLoss(mode="multiclass", ignore_index=ignore_index)

        metrics = MetricCollection(
            {
                "miou": MulticlassJaccardIndex(num_classes, ignore_index=ignore_index),
                "dice": MulticlassF1Score(num_classes, ignore_index=ignore_index),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation mIoU
        self.val_miou_best = MaxMetric()

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # make sure validation metrics don't store results from the sanity checks
        self.val_loss.reset()
        self.val_metrics.reset()
        self.val_miou_best.reset()

    def model_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input images and target masks.
        :return: A tuple containing (in order) the loss, the predicted masks and the target masks.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.ce_loss(logits, y)
        if self.hparams.dice_weight:
            loss = loss + self.hparams.dice_weight * self.dice_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input images and target masks.
        :param batch_idx: The index of the current batch.
        :return: The loss.
        """
        loss, preds, targets = self.model_step(batch)

        self.train_loss(loss)
        self.train_metrics(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input images and target masks.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        self.val_loss(loss)
        self.val_metrics(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        self.val_miou_best(self.val_metrics["miou"].compute())
        self.log("val/miou_best", self.val_miou_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input images and target masks.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        self.test_loss(loss)
        self.test_metrics(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        """Return predicted masks for a batch of images.

        :param batch: A batch whose first element is the input images.
        :param batch_idx: The index of the current batch.
        :return: A tensor of class indices of shape (B, H, W).
        """
        return self.forward(batch[0]).argmax(dim=1)


if __name__ == "__main__":
    _ = SegmentationLitModule(None, None, None, num_classes=2)
