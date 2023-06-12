from pathlib import Path

import torch
import numpy as np
from time import time
import os

from coupling_bound.emnist.data import make_dataloader
from coupling_bound.emnist.model import construct_net, layer_requires_grad

MODE_ITERATIVE = "iterative"
MODE_PROGRESSIVE = "progressive"
MODE_END_TO_END = "end-to-end"


def find_layers_requiring_grad(inn):
    layers_requiring_grad = []
    for layer_idx, layer in enumerate(inn.module_list):
        if layer_requires_grad(layer):
            layers_requiring_grad.append(layer_idx)
    return layers_requiring_grad


def stages_for_mode(inn, mode, n_epochs):
    all_layers = range(len(inn.module_list))
    layers_requiring_grad = find_layers_requiring_grad(inn)
    print(f"Found {len(layers_requiring_grad)} layers requiring gradient")
    if mode == MODE_END_TO_END:
        # End to end
        stages = [
            (all_layers, all_layers, n_epochs)
        ]
    else:
        # Iterative & progressive
        stages = []
        for layer in layers_requiring_grad:
            # active_layers = range(layer + 1)
            if mode == MODE_ITERATIVE:
                grad_layers = [layer]
            elif mode == MODE_PROGRESSIVE:
                grad_layers = range(layer + 1)
            else:
                raise ValueError(f"Mode {mode!r} not known.")
            stages.append((all_layers, grad_layers, n_epochs))
    return stages


class Trainer:
    def __init__(self, mode, max_depth, soft_permute, truncate_left,
                 fully_connected, all_in_one, subnet_width,
                 data_set, normalized, dequantization, data_root_dir,
                 lr, lr_schedule, n_epochs, batch_size,
                 verbose, epochs_per_line, save_frequency, device,
                 save_dir=None, logger=None):
        super().__init__()

        self.mode = mode
        self.max_depth = max_depth

        self.n_epochs = n_epochs
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.batch_size = batch_size

        self.epochs_per_line = epochs_per_line
        self.save_frequency = save_frequency
        self.verbose = verbose

        self.device = device

        if save_dir is None:
            self.save_dir = Path('./emnist_save/') / str(int(time()))
        else:
            self.save_dir = save_dir
        assert not self.save_dir.exists(), f"{self.save_dir} exists!"
        self.save_dir.mkdir(parents=True)

        self.data_root_dir = data_root_dir

        self.net = construct_net(data_set, max_depth, soft_permute,
                                 truncate_left, fully_connected, all_in_one,
                                 subnet_width)
        self.train_loader = make_dataloader(
            data_set, normalized=normalized, dequantization=dequantization,
            batch_size=self.batch_size, train=True, root_dir=self.data_root_dir
        )
        self.test_loader = make_dataloader(
            data_set, normalized=normalized, dequantization=dequantization,
            batch_size=1000, train=False, root_dir=self.data_root_dir,
            drop_last=True
        )
        self.n_dims = np.prod(self.net.dims_in)

        self.logger = logger

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def train_model(self):
        (self.save_dir / "model_save").mkdir()

        print(f'\nTraining model … \n')
        self.net.train()
        self.net.to(self.device)
        print(
            'time     stage     epoch    iteration         loss       last '
            'checkpoint')

        stages = stages_for_mode(self.net, self.mode, self.n_epochs)

        test_loss = None
        losses = []
        t0 = time()
        for stage_idx, (
                active_layers, grad_layers, n_epochs
        ) in enumerate(stages):
            self.net.set_active_layers(active_layers)
            self.net.set_grad_layers(grad_layers)

            optim = torch.optim.Adam(self.net.grad_parameters(), self.lr)
            sched = torch.optim.lr_scheduler.MultiStepLR(optim,
                                                         self.lr_schedule)

            for epoch in range(n_epochs):
                for batch_idx, (data, _) in enumerate(self.train_loader):
                    optim.zero_grad()

                    # Pass through net
                    data = data.to(self.device, non_blocking=True)
                    z, jac = self.net(data)  # latent space variable

                    # negative log-likelihood for gaussian in latent space
                    loss = torch.mean(z ** 2) / 2 - jac.mean() / self.n_dims
                    loss.backward()
                    optim.step()

                    # Logging
                    losses.append(loss.item())
                    if self.verbose:
                        self.print_loss(loss.item(), batch_idx, epoch, n_epochs,
                                        stage_idx, len(stages), t0)

                if (epoch + 1) % self.epochs_per_line == 0:
                    avg_loss = np.mean(losses)
                    self.print_loss(avg_loss, batch_idx, epoch, n_epochs,
                                    stage_idx, len(stages), t0,
                                    new_line=True)

                    with torch.no_grad():
                        self.net.eval()
                        test_losses = []
                        for test_data, _ in self.test_loader:
                            z, jac = self.net(
                                test_data.to(self.device, non_blocking=True)
                            )
                            test_losses.append((torch.mean(
                                z ** 2) / 2 - jac.mean() / self.n_dims
                                                ).item())
                        test_loss = sum(test_losses) / len(test_losses)
                        self.net.train()
                    self.print_loss(test_loss, batch_idx, epoch, n_epochs,
                                    stage_idx, len(stages), t0, new_line=True)

                    losses = []
                    if self.logger is not None:
                        self.logger.log_scalar("training.loss", avg_loss, epoch)
                        self.logger.log_scalar("test.loss", test_loss, epoch)
                if (epoch + 1) % self.save_frequency == 0:
                    self.save(os.path.join(self.save_dir, 'model_save',
                                           f'{stage_idx:02d}-{epoch + 1:03d}-{batch_idx + 1:04d}.pt'))

                sched.step()
        return test_loss

    def print_loss(self, loss, batch_idx, epoch, n_epochs, stage, n_stages, t0,
                   new_line=False):
        n_batches = len(self.train_loader)
        time_mins = (time() - t0) / 60
        print_str = f'  {time_mins :5.1f}   {stage + 1:03d}/{n_stages:03d}   {epoch + 1:03d}/{n_epochs:03d}   {batch_idx + 1:04d}/{n_batches:04d}   {loss:12.4f}'
        if new_line:
            print(print_str + ' ' * 40)
        else:
            last_save = (epoch // self.save_frequency) * self.save_frequency
            if last_save != 0:
                print_str += f'           {last_save:03d}'
            print(print_str, end='\r')

    def save(self, fname):
        torch.save({
            'model': self.net.state_dict()
        }, fname)

    def load(self, fname):
        self.net.load_state_dict(torch.load(fname)['model'])
