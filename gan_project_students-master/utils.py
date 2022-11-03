import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os
import json
from torch import Tensor
from typing import Tuple
from models import *
import copy


def load_gan(config: dict) -> Tuple[nn.Module, nn.Module]:
    """
    Parameters:
        config (dict), should contain:
            model (str): dcgan, wgan, wgan-gp, lsgan
            discriminator (str): path to discriminator weights
            generator (str): path to generator weights
            device (torch.device)
    """
    if config['model'] == 'dcgan':
        discriminator = Discriminator()
        generator = Generator(config)
    else:
        discriminator = Critic()
        generator = Generator(config)

    discriminator.to(config["device"])
    generator.to(config["device"])

    discriminator.load_state_dict(torch.load(config['discriminator'], map_location=config["device"]))
    generator.load_state_dict(torch.load(config['generator'], map_location=config["device"]))
    return generator, discriminator


def weights_init(model: nn.Module):
    """
    custom weights initialization called on gen and disc model

    """
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(model.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(model.weight, 1.0, 0.02)
        torch.nn.init.zeros_(model.bias)


def generator_loss(loss_fnc,
                   fake_output: torch.Tensor,
                   label: torch.Tensor) -> torch.Tensor:
    gen_loss = loss_fnc(fake_output, label)
    return gen_loss


def discriminator_loss(loss_fnc,
                       output: torch.Tensor, label: torch.Tensor):
    disc_loss = loss_fnc(output, label)
    return disc_loss


def show_images(images: torch.Tensor, denormalize=None):
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_xticks([])
    ax.set_yticks([])
    if denormalize:
        ax.imshow(make_grid(denormalize(images).detach(), nrow=22).permute(1, 2, 0))
    else:
        ax.imshow(make_grid(images.detach(), nrow=22).permute(1, 2, 0))
    plt.show()


def load_config(config: dict) -> dict:
    path = os.path.join(config["save_path"], config["name"], "config.json")
    with open(path) as file:
        config = json.load(file)
    return config


def save_config(config: dict):
    to_save_config = copy.deepcopy(config)
    path = os.path.join(to_save_config["save_path"], to_save_config["name"])
    create_dir(path)
    if to_save_config.get("device") is not None:
        del to_save_config["device"]
    with open(os.path.join(path, "config.json"), 'w') as file:
        json.dump(to_save_config, file)


def create_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


class Denormalize(torch.nn.Module):
    """Denormalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Args:
            tensor (Tensor): Tensor image to be denormalised.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))

        if not tensor.is_floating_point():
            raise TypeError('Input tensor should be a float tensor. Got {}.'.format(tensor.dtype))

        if tensor.ndim < 3:
            raise ValueError('Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = '
                             '{}.'.format(tensor.size()))

        if not self.inplace:
            tensor = tensor.clone()

        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        if (std == 0).any():
            raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        tensor.mul_(std).add_(mean)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
