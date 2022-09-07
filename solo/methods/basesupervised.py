# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from ast import Pass
from copy import deepcopy
from multiprocessing.spawn import prepare
import warnings
from argparse import ArgumentParser
from functools import partial
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from solo.utils.backbones import (
    convnext_base,
    convnext_large,
    convnext_small,
    convnext_tiny,
    poolformer_m36,
    poolformer_m48,
    poolformer_s12,
    poolformer_s24,
    poolformer_s36,
    swin_base,
    swin_large,
    swin_small,
    swin_tiny,
    vit_base,
    vit_large,
    vit_small,
    vit_tiny,
)
from solo.utils.knn import WeightedKNNClassifier
from solo.utils.lars import LARSWrapper
from solo.utils.metrics import accuracy_at_k, weighted_mean
from solo.utils.momentum import MomentumUpdater, initialize_momentum_params
######################################################
# Loading the data directly from the dataloader
from solo.utils.classification_dataloader import prepare_data as prepare_data_classification
######################################################
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet50

# Inserted modules originally not in the base code
import ipdb
from tqdm import tqdm
from copy import deepcopy

# Load various PaI methods (Rand, Mag, SNIP, GraSP, SynFlow)
## Import this later when everything works like a charm!
# from Pruners import pruners

def static_lr(
    get_lr: Callable, param_group_indexes: Sequence[int], lrs_to_replace: Sequence[float]
):
    lrs = get_lr()
    for idx, lr in zip(param_group_indexes, lrs_to_replace):
        lrs[idx] = lr
    return lrs


class BaseSupervisedMethod(pl.LightningModule):

    _SUPPORTED_BACKBONES = {
        "resnet18": resnet18,
        "resnet50": resnet50,
        "vit_tiny": vit_tiny,
        "vit_small": vit_small,
        "vit_base": vit_base,
        "vit_large": vit_large,
        "swin_tiny": swin_tiny,
        "swin_small": swin_small,
        "swin_base": swin_base,
        "swin_large": swin_large,
        "poolformer_s12": poolformer_s12,
        "poolformer_s24": poolformer_s24,
        "poolformer_s36": poolformer_s36,
        "poolformer_m36": poolformer_m36,
        "poolformer_m48": poolformer_m48,
        "convnext_tiny": convnext_tiny,
        "convnext_small": convnext_small,
        "convnext_base": convnext_base,
        "convnext_large": convnext_large,
    }

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        backbone_args: dict,
        max_epochs: int,
        batch_size: int,
        optimizer: str,
        lars: bool,
        lr: float,
        weight_decay: float,
        classifier_lr: float,
        exclude_bias_n_norm: bool,
        accumulate_grad_batches: Union[int, None],
        extra_optimizer_args: Dict,
        scheduler: str,
        num_large_crops: int,
        num_small_crops: int,
        min_lr: float = 0.0,
        warmup_start_lr: float = 0.00003,
        warmup_epochs: float = 10,
        scheduler_interval: str = "step",
        eta_lars: float = 1e-3,
        grad_clip_lars: bool = False,
        lr_decay_steps: Sequence = None,
        knn_eval: bool = False,
        knn_k: int = 20,
        sparsity: float = 0.1,
        pruner: str = "none",
        scope: str = "global",
        **kwargs,
    ):
        """Base model that implements all basic operations for all self-supervised methods.
        It adds shared arguments, extract basic learnable parameters, creates optimizers
        and schedulers, implements basic training_step for any number of crops,
        trains the online classifier and implements validation_step.

        Args:
            backbone (str): architecture of the base backbone.
            num_classes (int): number of classes.
            backbone_params (dict): dict containing extra backbone args, namely:
                #! optional, if it's not present, it is considered as False
                cifar (bool): flag indicating if cifar is being used.
                #! only for resnet
                zero_init_residual (bool): change the initialization of the resnet backbone.
                #! only for vit
                patch_size (int): size of the patches for ViT.
            max_epochs (int): number of training epochs.
            batch_size (int): number of samples in the batch.
            optimizer (str): name of the optimizer.
            lars (bool): flag indicating if lars should be used.
            lr (float): learning rate.
            weight_decay (float): weight decay for optimizer.
            classifier_lr (float): learning rate for the online linear classifier.
            exclude_bias_n_norm (bool): flag indicating if bias and norms should be excluded from
                lars.
            accumulate_grad_batches (Union[int, None]): number of batches for gradient accumulation.
            extra_optimizer_args (Dict): extra named arguments for the optimizer.
            scheduler (str): name of the scheduler.
            num_large_crops (int): number of big crops.
            num_small_crops (int): number of small crops .
            min_lr (float): minimum learning rate for warmup scheduler. Defaults to 0.0.
            warmup_start_lr (float): initial learning rate for warmup scheduler.
                Defaults to 0.00003.
            warmup_epochs (float): number of warmup epochs. Defaults to 10.
            scheduler_interval (str): interval to update the lr scheduler. Defaults to 'step'.
            eta_lars (float): eta parameter for lars.
            grad_clip_lars (bool): whether to clip the gradients in lars.
            lr_decay_steps (Sequence, optional): steps to decay the learning rate if scheduler is
                step. Defaults to None.
            knn_eval (bool): enables online knn evaluation while training.
            knn_k (int): the number of neighbors to use for knn.

        .. note::
            When using distributed data parallel, the batch size and the number of workers are
            specified on a per process basis. Therefore, the total batch size (number of workers)
            is calculated as the product of the number of GPUs with the batch size (number of
            workers).

        .. note::
            The learning rate (base, min and warmup) is automatically scaled linearly based on the
            batch size and gradient accumulation.

        .. note::
            For CIFAR10/100, the first convolutional and maxpooling layers of the ResNet backbone
            are slightly adjusted to handle lower resolution images (32x32 instead of 224x224).

        """

        super().__init__()

        # resnet backbone related
        self.backbone_args = backbone_args

        # training related
        self.num_classes = num_classes
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lars = lars
        self.lr = lr
        self.weight_decay = weight_decay
        self.classifier_lr = classifier_lr
        self.exclude_bias_n_norm = exclude_bias_n_norm
        self.accumulate_grad_batches = accumulate_grad_batches
        self.extra_optimizer_args = extra_optimizer_args
        self.scheduler = scheduler
        self.lr_decay_steps = lr_decay_steps
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.warmup_epochs = warmup_epochs
        assert scheduler_interval in ["step", "epoch"]
        self.scheduler_interval = scheduler_interval
        self.num_large_crops = num_large_crops
        self.num_small_crops = num_small_crops
        self.eta_lars = eta_lars
        self.grad_clip_lars = grad_clip_lars
        self.knn_eval = knn_eval
        self.knn_k = knn_k
        self._num_training_steps = None
        self.pruning_mask = {}
        self.sparsity = sparsity
        self.pruner = pruner
        self.scope = scope

        # multicrop
        self.num_crops = self.num_large_crops + self.num_small_crops

        # all the other parameters
        self.extra_args = kwargs

        # turn on multicrop if there are small crops
        self.multicrop = self.num_small_crops != 0

        # if accumulating gradient then scale lr
        if self.accumulate_grad_batches:
            self.lr = self.lr * self.accumulate_grad_batches
            self.classifier_lr = self.classifier_lr * self.accumulate_grad_batches
            self.min_lr = self.min_lr * self.accumulate_grad_batches
            self.warmup_start_lr = self.warmup_start_lr * self.accumulate_grad_batches

        assert backbone in BaseSupervisedMethod._SUPPORTED_BACKBONES
        self.base_model = self._SUPPORTED_BACKBONES[backbone]

        self.backbone_name = backbone

        # initialize backbone
        kwargs = self.backbone_args.copy()
        cifar = kwargs.pop("cifar", False)
        # swin specific
        if "swin" in self.backbone_name and cifar:
            kwargs["window_size"] = 4

        self.backbone = self.base_model(**kwargs)
        if "resnet" in self.backbone_name:
            self.features_dim = self.backbone.inplanes
            # remove fc layer
            self.backbone.fc = nn.Identity()
            if cifar:
                self.backbone.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=2, bias=False
                )
                self.backbone.maxpool = nn.Identity()
        else:
            self.features_dim = self.backbone.num_features

        self.classifier = nn.Linear(self.features_dim, num_classes)

        if self.pruner.lower != "none":
            # Input the pruning modules and schedules here
            ## PUT THE PRUNING MODULES HERE

            # Perform PaI right here
            # Iterate through parameters within the network.
            """
            There are two possible ways to iterate thorughout the network and check its prunability: 
            (1) First way is by referring the specific layer name and its parameter with named_parameters() and check the parameter names with it.
            i=0
            for name, p in resnet18().named_parameters():
                print('{} {}: {}'.format(i,name,p.shape))
                i+=1
            (2) Second way is by referring directly to the parameters with named_modules() and check if they're prunable
            i=0
            for name, module in (resnet18().named_modules()):
                if isinstance(module,(nn.Conv2d, nn.Linear)):
                    print(i,module)
                    i+=1
            We proceed with the first step
            """
            # Dataloader initialization (the one for gradient-based methods!)
            train_loader, val_loader = prepare_data_classification(
                dataset="cifar100",
                # data_dir="~/workspace/datasets/"
                train_dir="../../../datasets/cifar100/train",
                val_dir="../../../datasets/cifar100/val",
                batch_size=256,
                download=False,
                )
            device = torch.device("cuda:2")
            self.backbone.eval()
            # Perform all the gradient calculation here (if necessary!)
            if self.pruner.lower() == "snip":
                ## Compute gradient
                for data, target in tqdm(train_loader, desc="SNIP loss calculation"):
                    data, target = data.to(device), target.to(device)
                    self.backbone.to(device)
                    output = self.backbone(data)
                    loss = F.cross_entropy(output, target)
                    loss.backward()
                ## Calculate Score
                for name1, module in (self.backbone.named_modules()):
                    if isinstance(module, (nn.Conv2d, nn.Linear)):
                        for name2, param in (module.named_parameters(recurse=False)):
                            key = name1 + '.' + name2
                            # Assign scores to the respective parameters here
                            self.pruning_mask[key] = torch.clone(param.grad.cpu()).detach().abs_()           
                            # Set gradients to zero
                            param.grad.data.zero_()
                    else:
                        for name2, param in (module.named_parameters(recurse=False)):
                            key = name1 + '.' + name2
                            # Set gradients to zero
                            param.grad.data.zero_()

                # Normalize the gradients
                all_scores = torch.cat([torch.flatten(v) for v in self.pruning_mask.values()])
                norm = torch.sum(all_scores)
                for keys in self.pruning_mask.keys():
                    self.pruning_mask[keys].div_(norm)

            elif self.pruner.lower() == "grasp":
                temp, eps = 200, 1e-10
                # First, evaluate gradient vector without computational graph
                stopped_grads = 0
                for data, target in tqdm(train_loader, desc="GraSP 1st loss calculation"):
                    data, target = data.to(device), target.to(device)
                    self.backbone.to(device)
                    output = self.backbone(data) / temp
                    L = F.cross_entropy(output, target)
                    
                    grads = torch.autograd.grad(L, [param for _, param in self.backbone.named_parameters()], create_graph=False)
                    flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])
                    stopped_grads += flatten_grads

                # Next, evaluate gradient vector with computational graph
                for data, target in tqdm(train_loader, desc="GraSP 2nd loss calculation"):
                    data, target = data.to(device), target.to(device)
                    self.backbone.to(device)
                    output = self.backbone(data) / temp
                    L = F.cross_entropy(output, target)

                    grads = torch.autograd.grad(L, [param for _, param in self.backbone.named_parameters()], create_graph=True)
                    flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])

                    gnorm = (stopped_grads * flatten_grads).sum()
                    gnorm.backward()

                # Calculate the score
                for name1, module in (self.backbone.named_modules()):
                    if isinstance(module, (nn.Conv2d, nn.Linear)):
                        for name2, param in (module.named_parameters(recurse=False)):
                            key = name1 + '.' + name2
                            # Assign scores to the respective parameters here
                            self.pruning_mask[key] = torch.clone(param.grad.cpu() * param.cpu()).detach()
                            # Set gradients to zero
                            param.grad.data.zero_()
                    else:
                        for name2, param in (module.named_parameters(recurse=False)):
                            param.grad.data.zero_()

                # Normalize the gradients
                all_scores = torch.cat([torch.flatten(v) for v in self.pruning_mask.values()])
                norm = torch.sum(all_scores)
                for keys in self.pruning_mask.keys():
                    self.pruning_mask[keys].div_(norm)
                # ipdb.set_trace()

            elif self.pruner.lower() == "synflow":
                """
                As far as I'm concerned, SynFlow is multi-schedule Pruning method;
                This means that pruning is done not only once, but multiple times concurrently for 200 epochs
                """

                # Assign parameters to its absolute value and save the signs in a dict "sign".
                @torch.no_grad()
                def linearize(model):
                    signs = {}
                    for name, param in model.state_dict().items():
                        signs[name] = torch.sign(param)
                        param.abs_()
                    return signs
                
                # Restore the sign parameters into the distilled params
                @torch.no_grad()
                def nonlinearlize(model, signs):
                    for name, param in model.state_dict().items():
                        param.mul_(signs[name])

                signs = linearize(self.backbone)

                # Pruning gradually by 200 epochs and exponential scheduler
                prune_epochs = 200
                ######################################################################
                (data, _) = next(iter(train_loader))
                input_dim = list(data[0,:].shape)
                input = torch.ones([1] + input_dim).to(device)
                self.backbone.to(device)
                
                # Save the model's original state
                original_state = deepcopy(self.backbone.state_dict())

                for epoch in range(prune_epochs):
                    # Check if pruning mask is empty or not
                    if self.pruning_mask:
                        # Generate the mask obtained through global thresholding
                        sparseness = self.sparsity**((epoch + 1) / prune_epochs)
                        global_scores = torch.cat([torch.flatten(self.pruning_mask[v]) for v in self.pruning_mask])
                        k = int((1.0 - sparseness) * global_scores.numel())
                        if not k < 1:
                            threshold, _ = torch.kthvalue(torch.flatten(global_scores), k)
                            for name1, module in (self.backbone.named_modules()):
                                if isinstance(module, (nn.Conv2d, nn.Linear)):
                                    # Iterate through parameters in prunable modules.
                                    for name2, param in module.named_parameters(recurse=False):
                                        key = name1 + '.' + name2
                                        score = self.pruning_mask[key]
                                        # Now, set parameters to be pruned and one that should be left as it is
                                        self.pruning_mask[key] = torch.where(score <= threshold, 0, 1)
                        # Apply the generated mask towards the backbone
                        for module_name, param in self.backbone.named_parameters():
                            if module_name in self.pruning_mask.keys():
                                param.data = self.pruning_mask[module_name].to(param.data.get_device()) * param.data

                    # Now, apply the backbone masking around here
                    output = self.backbone(input)
                    torch.sum(output).backward()

                    # Calculate the score
                    for name1, module in (self.backbone.named_modules()):
                        if isinstance(module, (nn.Conv2d, nn.Linear)):
                            for name2, param in (module.named_parameters(recurse=False)):
                                key = name1 + '.' + name2
                                # Assign scores to the respective parameters here
                                self.pruning_mask[key] = torch.clone(param.grad.cpu() * param.cpu()).detach().abs_()
                                # Set gradients to zero
                                param.grad.data.zero_()
                        else:
                            for name2, param in (module.named_parameters(recurse=False)):
                                param.grad.data.zero_()

                ######################################################################

                # Revert back to the original state
                self.backbone.load_state_dict(deepcopy(original_state))
                self.backbone.to("cpu")
                # Multiply the originally saved state onto the current state
                nonlinearlize(self.backbone, signs)
                
            elif self.pruner.lower() in ["rand","mag"]:
                # Iterate through modules in backbone
                for name1, module in (self.backbone.named_modules()):
                    ## Proceed if the module within this specific network is prunable (conv2d, linear)
                    if isinstance(module, (nn.Conv2d, nn.Linear)):
                        ## Iterate through paramaters in prunable modules.
                        for name2, param in module.named_parameters(recurse=False):
                            # Create masks for each one of them
                            key = name1 + '.' + name2
                            self.pruning_mask[key] = param
                            ## Now that we successfully masked each one of them, it's time to score parameters
                            ## (1) This one is for rand pruning
                            if self.pruner.lower() == "rand":
                                self.pruning_mask[key] = torch.randn_like(self.pruning_mask[key])
                            ## (2) This one is for magnitude pruning (essentially, return the absolute value of the magnitude)
                            elif self.pruner.lower() == "mag":
                                self.pruning_mask[key] = torch.clone(self.pruning_mask[key]).detach().abs_()
            else:
                raise ValueError(f'{self.pruner} not in (rand, mag, snip, grasp, synflow)')

            ### Now, perform thresholding towards the parameters' score.
            
            ## This is for layer-wise pruning
            if self.scope.lower() == "layer":
                for name1, module in (self.backbone.named_modules()):
                    ## Proceed if the module within this specific network is prunable (conv2d, linear)
                    if isinstance(module, (nn.Conv2d, nn.Linear)):
                        ## Iterate through paramaters in prunable modules.
                        for name2, param in module.named_parameters(recurse=False):
                            key = name1 + '.' + name2
                            score = self.pruning_mask[key]
                            k = int((1.0 - self.sparsity) * score.numel())
                            if not k < 1:
                                ## Perform thresholding under the sane case
                                threshold, _ = torch.kthvalue(torch.flatten(score), k)
                                ## Now, set parameters to be pruned and one that should be left as it is
                                self.pruning_mask[key] = torch.where(score <= threshold, 0, 1)

            ## This is for global-wise pruning (Default Case)
            elif self.scope.lower() == "global":
                global_scores = torch.cat([torch.flatten(self.pruning_mask[v]) for v in self.pruning_mask])
                # Find the threshold of pruning
                k = int((1.0 - self.sparsity) * global_scores.numel())
                if not k < 1:
                    threshold, _ = torch.kthvalue(torch.flatten(global_scores), k)
                    for name1, module in (self.backbone.named_modules()):
                        if isinstance(module, (nn.Conv2d, nn.Linear)):
                            # Iterate through parameters in prunable modules.
                            for name2, param in module.named_parameters(recurse=False):
                                key = name1 + '.' + name2
                                score = self.pruning_mask[key]
                                # Now, set parameters to be pruned and one that should be left as it is
                                self.pruning_mask[key] = torch.where(score <= threshold, 0, 1)
                                
            print("\nPruning finished!\n")

        if self.knn_eval:
            self.knn = WeightedKNNClassifier(k=self.knn_k, distance_fx="euclidean")

        if scheduler_interval == "step":
            warnings.warn(
                f"Using scheduler_interval={scheduler_interval} might generate "
                "issues when resuming a checkpoint."
            )

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds shared basic arguments that are shared for all methods.

        Args:
            parent_parser (ArgumentParser): argument parser that is used to create a
                argument group.

        Returns:
            ArgumentParser: same as the argument, used to avoid errors.
        """

        parser = parent_parser.add_argument_group("base")

        # backbone args
        SUPPORTED_BACKBONES = BaseSupervisedMethod._SUPPORTED_BACKBONES

        parser.add_argument("--backbone", choices=SUPPORTED_BACKBONES, type=str)
        # extra args for resnet
        parser.add_argument("--zero_init_residual", action="store_true")
        # extra args for ViT
        parser.add_argument("--patch_size", type=int, default=16)

        # general train
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--lr", type=float, default=0.3)
        parser.add_argument("--classifier_lr", type=float, default=0.3)
        parser.add_argument("--weight_decay", type=float, default=0.0001)
        parser.add_argument("--num_workers", type=int, default=4)

        # wandb
        parser.add_argument("--name")
        parser.add_argument("--project")
        parser.add_argument("--entity", default=None, type=str)
        parser.add_argument("--wandb", action="store_true")
        parser.add_argument("--offline", action="store_true")

        # optimizer
        SUPPORTED_OPTIMIZERS = ["sgd", "adam", "adamw"]

        parser.add_argument("--optimizer", choices=SUPPORTED_OPTIMIZERS, type=str, required=True)
        parser.add_argument("--lars", action="store_true")
        parser.add_argument("--grad_clip_lars", action="store_true")
        parser.add_argument("--eta_lars", default=1e-3, type=float)
        parser.add_argument("--exclude_bias_n_norm", action="store_true")

        # scheduler
        SUPPORTED_SCHEDULERS = [
            "reduce",
            "warmup_cosine",
            "step",
            "exponential",
            "none",
        ]

        parser.add_argument("--scheduler", choices=SUPPORTED_SCHEDULERS, type=str, default="reduce")
        parser.add_argument("--lr_decay_steps", default=None, type=int, nargs="+")
        parser.add_argument("--min_lr", default=0.0, type=float)
        parser.add_argument("--warmup_start_lr", default=0.00003, type=float)
        parser.add_argument("--warmup_epochs", default=10, type=int)
        parser.add_argument(
            "--scheduler_interval", choices=["step", "epoch"], default="step", type=str
        )

        # DALI only
        # uses sample indexes as labels and then gets the labels from a lookup table
        # this may use more CPU memory, so just use when needed.
        parser.add_argument("--encode_indexes_into_labels", action="store_true")

        # online knn eval
        parser.add_argument("--knn_eval", action="store_true")
        parser.add_argument("--knn_k", default=20, type=int)

        # Add parser for the PaI methods
        SUPPORTED_PRUNERS = [
            "rand",
            "mag",
            "snip",
            "grasp",
            "synflow",
            "none"
        ]
        parser.add_argument("--pruner", choices=SUPPORTED_PRUNERS, type=str, default="none")
        parser.add_argument("--sparsity", default=0.1, type=float)

        SUPPORTED_SCOPES = [
            "global",
            "layer"
        ]
        parser.add_argument("--scope", choices=SUPPORTED_SCOPES, type=str, default="global")

        return parent_parser

    def set_loaders(self, train_loader: DataLoader = None, val_loader: DataLoader = None) -> None:
        """Sets dataloaders so that you can obtain extra information about them.
        We currently only use to obtain the number of training steps per epoch.

        Args:
            train_loader (DataLoader, optional): training dataloader.
            val_loader (DataLoader, optional): validation dataloader.

        """

        if train_loader is not None:
            self.train_dataloader = lambda: train_loader

        if val_loader is not None:
            self.val_dataloader = lambda: val_loader

    @property
    def num_training_steps(self) -> int:
        """Compute the number of training steps for each epoch."""

        if self._num_training_steps is None:
            if self.trainer.train_dataloader is None:
                try:
                    dataloader = self.train_dataloader()
                except NotImplementedError:
                    raise RuntimeError(
                        "To use linear warmup cosine annealing lr"
                        "set the dataloader with .set_loaders(...)"
                    )

            dataset_size = getattr(self, "dali_epoch_size", None) or len(dataloader.dataset)

            dataset_size = self.trainer.limit_train_batches * dataset_size

            num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)

            if self.trainer.tpu_cores:
                num_devices = max(num_devices, self.trainer.tpu_cores)

            effective_batch_size = (
                self.batch_size * self.trainer.accumulate_grad_batches * num_devices
            )
            self._num_training_steps = dataset_size // effective_batch_size

        return self._num_training_steps

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Defines learnable parameters for the base class.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """

        return [
            {"name": "backbone", "params": self.backbone.parameters()},
            {
                "name": "classifier",
                "params": self.classifier.parameters(),
                "lr": self.classifier_lr,
                "weight_decay": 0,
            },
        ]

    # Configure the pruners

    def configure_optimizers(self) -> Tuple[List, List]:
        """Collects learnable parameters and configures the optimizer and learning rate scheduler.

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        # collect learnable parameters
        idxs_no_scheduler = [
            i for i, m in enumerate(self.learnable_params) if m.pop("static_lr", False)
        ]

        # select optimizer
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam
        elif self.optimizer == "adamw":
            optimizer = torch.optim.AdamW
        else:
            raise ValueError(f"{self.optimizer} not in (sgd, adam, adamw)")

        # create optimizer
        optimizer = optimizer(
            self.learnable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )
        # optionally wrap with lars
        if self.lars:
            assert self.optimizer == "sgd", "LARS is only compatible with SGD."
            optimizer = LARSWrapper(
                optimizer,
                eta=self.eta_lars,
                clip=self.grad_clip_lars,
                exclude_bias_n_norm=self.exclude_bias_n_norm,
            )

        if self.scheduler == "none":
            return optimizer

        if self.scheduler == "warmup_cosine":
            scheduler = {
                "scheduler": LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=self.warmup_epochs * self.num_training_steps,
                    max_epochs=self.max_epochs * self.num_training_steps,
                    warmup_start_lr=self.warmup_start_lr,
                    eta_min=self.min_lr,
                ),
                "interval": self.scheduler_interval,
                "frequency": 1,
            }
        elif self.scheduler == "step":
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps)
        else:
            raise ValueError(f"{self.scheduler} not in (warmup_cosine, cosine, step)")

        if idxs_no_scheduler:
            partial_fn = partial(
                static_lr,
                get_lr=scheduler["scheduler"].get_lr
                if isinstance(scheduler, dict)
                else scheduler.get_lr,
                param_group_indexes=idxs_no_scheduler,
                lrs_to_replace=[self.lr] * len(idxs_no_scheduler),
            )
            if isinstance(scheduler, dict):
                scheduler["scheduler"].get_lr = partial_fn
            else:
                scheduler.get_lr = partial_fn

        return [optimizer], [scheduler]

    def forward(self, *args, **kwargs) -> Dict:
        """Dummy forward, calls base forward."""

        return self.base_forward(*args, **kwargs)

    def base_forward(self, X: torch.Tensor) -> Dict:
        """Basic forward that allows children classes to override forward().

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict: dict of logits and features.
        """
        for module_name, param in self.backbone.named_parameters():
            if module_name in self.pruning_mask.keys():
                param.data = self.pruning_mask[module_name].to(param.data.get_device()) * param.data
        ############### 
        feats = self.backbone(X)
        # logits = self.classifier(feats.detach())
        # In supervised setup, we don't detach the features.
        logits = self.classifier(feats)
        return {
            "logits": logits,
            "feats": feats,
        }

    def _base_shared_step(self, X: torch.Tensor, targets: torch.Tensor) -> Dict:
        """Forwards a batch of images X and computes the classification loss, the logits, the
        features, acc@1 and acc@5.

        Args:
            X (torch.Tensor): batch of images in tensor format.
            targets (torch.Tensor): batch of labels for X.

        Returns:
            Dict: dict containing the classification loss, logits, features, acc@1 and acc@5.
        """

        out = self.base_forward(X)
        logits = out["logits"]

        loss = F.cross_entropy(logits, targets, ignore_index=-1)
        # handle when the number of classes is smaller than 5
        top_k_max = min(5, logits.size(1))
        acc1, acc5 = accuracy_at_k(logits, targets, top_k=(1, top_k_max))

        return {**out, "loss": loss, "acc1": acc1, "acc5": acc5}

    def training_step(self, batch: List[Any], batch_idx: int) -> Dict[str, Any]:
        """Training step for pytorch lightning. It does all the shared operations, such as
        forwarding the crops, computing logits and computing statistics.

        Args:
            batch (List[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            Dict[str, Any]: dict with the classification loss, features and logits.
        """

        _, X, targets = batch

        X = [X] if isinstance(X, torch.Tensor) else X

        # check that we received the desired number of crops
        assert len(X) == self.num_crops

        outs = [self._base_shared_step(x, targets) for x in X[: self.num_large_crops]]
        outs = {k: [out[k] for out in outs] for k in outs[0].keys()}

        if self.multicrop:
            outs["feats"].extend([self.backbone(x) for x in X[self.num_large_crops :]])

        # Calculate the sparsity of the network
        nonzero_param, total_param = 0, 0
        for module_name, param in self.backbone.named_modules():
            ## Proceed if the module within this specific network is prunable (conv2d, linear)
            if isinstance(param, (nn.Conv2d, nn.Linear)) and module_name != 'fc':
                nonzero_param += torch.count_nonzero(param.weight)
                total_param += torch.numel(param.weight)

        # Create logs and stats for sparsity
        sparsity_log = {
            "sparsity": nonzero_param/total_param,
        }
        # print("\nSparsity Calculated\n")
        self.log_dict(sparsity_log, on_epoch=True, sync_dist=True)

        # loss and stats
        outs["loss"] = sum(outs["loss"]) / self.num_large_crops
        outs["acc1"] = sum(outs["acc1"]) / self.num_large_crops
        outs["acc5"] = sum(outs["acc5"]) / self.num_large_crops

        metrics = {
            "train_class_loss": outs["loss"],
            "train_acc1": outs["acc1"],
            "train_acc5": outs["acc5"],
        }

        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        if self.knn_eval:
            targets = targets.repeat(self.num_large_crops)
            mask = targets != -1
            self.knn(
                train_features=torch.cat(outs["feats"][: self.num_large_crops])[mask].detach(),
                train_targets=targets[mask],
            )

        return outs

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int, dataloader_idx: int = None
    ) -> Dict[str, Any]:
        """Validation step for pytorch lightning. It does all the shared operations, such as
        forwarding a batch of images, computing logits and computing metrics.

        Args:
            batch (List[torch.Tensor]):a batch of data in the format of [img_indexes, X, Y].
            batch_idx (int): index of the batch.

        Returns:
            Dict[str, Any]: dict with the batch_size (used for averaging), the classification loss
                and accuracies.
        """

        X, targets = batch
        batch_size = targets.size(0)

        out = self._base_shared_step(X, targets)

        if self.knn_eval and not self.trainer.sanity_checking:
            self.knn(test_features=out.pop("feats").detach(), test_targets=targets.detach())

        metrics = {
            "batch_size": batch_size,
            "val_loss": out["loss"],
            "val_acc1": out["acc1"],
            "val_acc5": out["acc5"],
        }
        return metrics

    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.

        Args:
            outs (List[Dict[str, Any]]): list of outputs of the validation step.
        """

        val_loss = weighted_mean(outs, "val_loss", "batch_size")
        val_acc1 = weighted_mean(outs, "val_acc1", "batch_size")
        val_acc5 = weighted_mean(outs, "val_acc5", "batch_size")

        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}

        if self.knn_eval and not self.trainer.sanity_checking:
            val_knn_acc1, val_knn_acc5 = self.knn.compute()
            log.update({"val_knn_acc1": val_knn_acc1, "val_knn_acc5": val_knn_acc5})

        self.log_dict(log, sync_dist=True)


class BaseMomentumMethod(BaseSupervisedMethod):
    def __init__(
        self,
        base_tau_momentum: float,
        final_tau_momentum: float,
        momentum_classifier: bool,
        **kwargs,
    ):
        """Base momentum model that implements all basic operations for all self-supervised methods
        that use a momentum backbone. It adds shared momentum arguments, adds basic learnable
        parameters, implements basic training and validation steps for the momentum backbone and
        classifier. Also implements momentum update using exponential moving average and cosine
        annealing of the weighting decrease coefficient.

        Args:
            base_tau_momentum (float): base value of the weighting decrease coefficient (should be
                in [0,1]).
            final_tau_momentum (float): final value of the weighting decrease coefficient (should be
                in [0,1]).
            momentum_classifier (bool): whether or not to train a classifier on top of the momentum
                backbone.
        """

        super().__init__(**kwargs)

        # momentum backbone
        kwargs = self.backbone_args.copy()
        cifar = kwargs.pop("cifar", False)
        # swin specific
        if "swin" in self.backbone_name and cifar:
            kwargs["window_size"] = 4

        self.momentum_backbone = self.base_model(**kwargs)
        if "resnet" in self.backbone_name:
            self.features_dim = self.momentum_backbone.inplanes
            # remove fc layer
            self.momentum_backbone.fc = nn.Identity()
            if cifar:
                self.momentum_backbone.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=2, bias=False
                )
                self.momentum_backbone.maxpool = nn.Identity()
        else:
            self.features_dim = self.momentum_backbone.num_features

        initialize_momentum_params(self.backbone, self.momentum_backbone)

        # momentum classifier
        if momentum_classifier:
            self.momentum_classifier: Any = nn.Linear(self.features_dim, self.num_classes)
        else:
            self.momentum_classifier = None

        # momentum updater
        self.momentum_updater = MomentumUpdater(base_tau_momentum, final_tau_momentum)

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Adds momentum classifier parameters to the parameters of the base class.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """

        momentum_learnable_parameters = []
        if self.momentum_classifier is not None:
            momentum_learnable_parameters.append(
                {
                    "name": "momentum_classifier",
                    "params": self.momentum_classifier.parameters(),
                    "lr": self.classifier_lr,
                    "weight_decay": 0,
                }
            )
        return super().learnable_params + momentum_learnable_parameters

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Defines base momentum pairs that will be updated using exponential moving average.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs (two element tuples).
        """

        return [(self.backbone, self.momentum_backbone)]

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds basic momentum arguments that are shared for all methods.

        Args:
            parent_parser (ArgumentParser): argument parser that is used to create a
                argument group.

        Returns:
            ArgumentParser: same as the argument, used to avoid errors.
        """

        parent_parser = super(BaseMomentumMethod, BaseMomentumMethod).add_model_specific_args(
            parent_parser
        )
        parser = parent_parser.add_argument_group("base")

        # momentum settings
        parser.add_argument("--base_tau_momentum", default=0.99, type=float)
        parser.add_argument("--final_tau_momentum", default=1.0, type=float)
        parser.add_argument("--momentum_classifier", action="store_true")

        return parent_parser

    def on_train_start(self):
        """Resets the step counter at the beginning of training."""
        self.last_step = 0

    @torch.no_grad()
    def base_momentum_forward(self, X: torch.Tensor) -> Dict:
        """Momentum forward that allows children classes to override how the momentum backbone is used.
        Args:
            X (torch.Tensor): batch of images in tensor format.
        Returns:
            Dict: dict of logits and features.
        """

        feats = self.momentum_backbone(X)
        return {"feats": feats}

    def _shared_step_momentum(self, X: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        """Forwards a batch of images X in the momentum backbone and optionally computes the
        classification loss, the logits, the features, acc@1 and acc@5 for of momentum classifier.

        Args:
            X (torch.Tensor): batch of images in tensor format.
            targets (torch.Tensor): batch of labels for X.

        Returns:
            Dict[str, Any]:
                a dict containing the classification loss, logits, features, acc@1 and
                acc@5 of the momentum backbone / classifier.
        """

        out = self.base_momentum_forward(X)

        if self.momentum_classifier is not None:
            feats = out["feats"]
            logits = self.momentum_classifier(feats)

            loss = F.cross_entropy(logits, targets, ignore_index=-1)
            acc1, acc5 = accuracy_at_k(logits, targets, top_k=(1, 5))
            out.update({"logits": logits, "loss": loss, "acc1": acc1, "acc5": acc5})

        return out

    def training_step(self, batch: List[Any], batch_idx: int) -> Dict[str, Any]:
        """Training step for pytorch lightning. It performs all the shared operations for the
        momentum backbone and classifier, such as forwarding the crops in the momentum backbone
        and classifier, and computing statistics.
        Args:
            batch (List[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            Dict[str, Any]: a dict with the features of the momentum backbone and the classification
                loss and logits of the momentum classifier.
        """

        outs = super().training_step(batch, batch_idx)

        _, X, targets = batch
        X = [X] if isinstance(X, torch.Tensor) else X

        # remove small crops
        X = X[: self.num_large_crops]

        momentum_outs = [self._shared_step_momentum(x, targets) for x in X]
        momentum_outs = {
            "momentum_" + k: [out[k] for out in momentum_outs] for k in momentum_outs[0].keys()
        }

        if self.momentum_classifier is not None:
            # momentum loss and stats
            momentum_outs["momentum_loss"] = (
                sum(momentum_outs["momentum_loss"]) / self.num_large_crops
            )
            momentum_outs["momentum_acc1"] = (
                sum(momentum_outs["momentum_acc1"]) / self.num_large_crops
            )
            momentum_outs["momentum_acc5"] = (
                sum(momentum_outs["momentum_acc5"]) / self.num_large_crops
            )

            metrics = {
                "train_momentum_class_loss": momentum_outs["momentum_loss"],
                "train_momentum_acc1": momentum_outs["momentum_acc1"],
                "train_momentum_acc5": momentum_outs["momentum_acc5"],
            }
            self.log_dict(metrics, on_epoch=True, sync_dist=True)

            # adds the momentum classifier loss together with the general loss
            outs["loss"] += momentum_outs["momentum_loss"]

        return {**outs, **momentum_outs}

    def on_train_batch_end(
        self, outputs: Dict[str, Any], batch: Sequence[Any], batch_idx: int, dataloader_idx: int
    ):
        """Performs the momentum update of momentum pairs using exponential moving average at the
        end of the current training step if an optimizer step was performed.

        Args:
            outputs (Dict[str, Any]): the outputs of the training step.
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.
            dataloader_idx (int): index of the dataloader.
        """

        if self.trainer.global_step > self.last_step:
            # update momentum backbone and projector
            momentum_pairs = self.momentum_pairs
            for mp in momentum_pairs:
                self.momentum_updater.update(*mp)
            # log tau momentum
            self.log("tau", self.momentum_updater.cur_tau)
            # update tau
            cur_step = self.trainer.global_step
            if self.trainer.accumulate_grad_batches:
                cur_step = cur_step * self.trainer.accumulate_grad_batches
            self.momentum_updater.update_tau(
                cur_step=cur_step,
                max_steps=len(self.trainer.train_dataloader) * self.trainer.max_epochs,
            )
        self.last_step = self.trainer.global_step

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int, dataloader_idx: int = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Validation step for pytorch lightning. It performs all the shared operations for the
        momentum backbone and classifier, such as forwarding a batch of images in the momentum
        backbone and classifier and computing statistics.
        Args:
            batch (List[torch.Tensor]): a batch of data in the format of [X, Y].
            batch_idx (int): index of the batch.
        Returns:
            Tuple(Dict[str, Any], Dict[str, Any]): tuple of dicts containing the batch_size (used
                for averaging), the classification loss and accuracies for both the online and the
                momentum classifiers.
        """

        parent_metrics = super().validation_step(batch, batch_idx)

        X, targets = batch
        batch_size = targets.size(0)

        out = self._shared_step_momentum(X, targets)

        metrics = None
        if self.momentum_classifier is not None:
            metrics = {
                "batch_size": batch_size,
                "momentum_val_loss": out["loss"],
                "momentum_val_acc1": out["acc1"],
                "momentum_val_acc5": out["acc5"],
            }

        return parent_metrics, metrics

    def validation_epoch_end(self, outs: Tuple[List[Dict[str, Any]]]):
        """Averages the losses and accuracies of the momentum backbone / classifier for all the
        validation batches. This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.
        Args:
            outs (Tuple[List[Dict[str, Any]]]):): list of outputs of the validation step for self
                and the parent.
        """

        parent_outs = [out[0] for out in outs]
        super().validation_epoch_end(parent_outs)

        if self.momentum_classifier is not None:
            momentum_outs = [out[1] for out in outs]

            val_loss = weighted_mean(momentum_outs, "momentum_val_loss", "batch_size")
            val_acc1 = weighted_mean(momentum_outs, "momentum_val_acc1", "batch_size")
            val_acc5 = weighted_mean(momentum_outs, "momentum_val_acc5", "batch_size")

            log = {
                "momentum_val_loss": val_loss,
                "momentum_val_acc1": val_acc1,
                "momentum_val_acc5": val_acc5,
            }
            self.log_dict(log, sync_dist=True)
