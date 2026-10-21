from collections.abc import Callable

import torch
from torchmetrics import MaxMetric, MeanMetric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from src.models.base_module import BaseLitModule


class ClassificationLitModule(BaseLitModule):
    """`LightningModule` for multiclass image classification.

    The network (e.g. any `timm` model), optimizer and scheduler are defined in the config, see
    `configs/model/classification.yaml`.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: Callable[..., torch.optim.Optimizer],
        scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler] | None,
        num_classes: int,
        label_smoothing: float = 0.0,
        compile: bool = False,
    ) -> None:
        """Initialize a `ClassificationLitModule`.

        :param net: The model to train, returning logits of shape (B, num_classes).
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param num_classes: The number of classes.
        :param label_smoothing: Label smoothing for the cross-entropy loss.
        :param compile: Whether to `torch.compile` the model.
        """
        super().__init__(net=net, optimizer=optimizer, scheduler=scheduler, compile=compile)

        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        metrics = MetricCollection(
            {
                "acc": MulticlassAccuracy(num_classes=num_classes, average="micro"),
                "f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_metrics.reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input images and target labels.
        :return: A tuple containing (in order) the loss, the predictions and the target labels.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input images and target labels.
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

        :param batch: A batch of data (a tuple) containing the input images and target labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        self.val_loss(loss)
        self.val_metrics(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        acc = self.val_metrics["acc"].compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input images and target labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        self.test_loss(loss)
        self.test_metrics(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        """Return class probabilities for a batch of images.

        :param batch: A batch whose first element is the input images.
        :param batch_idx: The index of the current batch.
        :return: A tensor of class probabilities of shape (B, num_classes).
        """
        return self.forward(batch[0]).softmax(dim=1)


if __name__ == "__main__":
    _ = ClassificationLitModule(None, None, None, num_classes=10)
