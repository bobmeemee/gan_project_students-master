import torch
import torch.nn as nn
from collections import OrderedDict


class Generator(nn.Module):
    """Your implementation of the generator of DCGAN"""

    def __init__(self, config: dict):

        super(Generator, self).__init__()
        self.main = nn.Sequential(
            OrderedDict([
                # Block 1:input is Z, going into a convolution
                ('ConvTranspose2d_1',
                 nn.ConvTranspose2d(config["latent_dim"], 64 * 8, kernel_size=4, stride=1, padding=0)),
                ('BatchNorm2d_1', nn.BatchNorm2d(64 * 8)),
                ('ReLU_1', nn.ReLU()),

                # Block 2: input is (64 * 8) x 4 x 4
                ('ConvTranspose2d_2',
                 nn.ConvTranspose2d(64 * 8, 64 * 4, kernel_size=4, stride=2, padding=1)),
                ('BatchNorm2d_2', nn.BatchNorm2d(64 * 4)),
                ('ReLU_2', nn.ReLU()),

                # Block 3: input is (64 * 4) x 8 x 8
                ('ConvTranspose2d_3',
                 nn.ConvTranspose2d(64 * 4, 64 * 2, kernel_size=4, stride=2, padding=1)),
                ('BatchNorm2d_3', nn.BatchNorm2d(64 * 2)),
                ('ReLU_3', nn.ReLU()),

                # Block 4: input is (64 * 2) x 16 x 16
                ('ConvTranspose2d_4',
                 nn.ConvTranspose2d(64 * 2, 64, kernel_size=4, stride=2, padding=1)),
                ('BatchNorm2d_4', nn.BatchNorm2d(64)),
                ('ReLU_4', nn.ReLU()),

                # Block 5: input is (64) x 32 x 32
                ('ConvTranspose2d_5',
                 nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)),
                ('Tanh',
                 nn.Tanh())
                # Output: output is (3) x 64 x 64
            ])
        )

    def forward(self, input: torch.tensor) -> torch.Tensor:
        output = self.main(input)
        return output


class Discriminator(nn.Module):
    """Your implementation of the discriminator of DCGAN"""

    def __init__(self):

        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            OrderedDict([
                # Block 1: input is (3) x 64 x 64
                ('Conv2d_1', nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)),
                ('LeakyReLU_1', nn.LeakyReLU()),

                # Block 2: input is (64) x 32 x 32
                ('Conv2d_2', nn.Conv2d(64, 64*2, kernel_size=4, stride=2, padding=1)),
                ('BatchNorm2d_2', nn.BatchNorm2d(64*2)),
                ('LeakyReLU_2', nn.LeakyReLU()),

                # Block 3: input is (64*2) x 16 x 16
                ('Conv2d_3', nn.Conv2d(64*2, 64*4, kernel_size=4, stride=2, padding=1)),
                ('BatchNorm2d_3', nn.BatchNorm2d(64*4)),
                ('LeakyReLU_3', nn.LeakyReLU()),

                # Block 4: input is (64*4) x 8 x 8
                ('Conv2d_4', nn.Conv2d(64*4, 64*8, kernel_size=4, stride=2, padding=1)),
                ('BatchNorm2d_4', nn.BatchNorm2d(64*8)),
                ('LeakyReLU_4', nn.LeakyReLU()),

                # Block 5: input is (64*8) x 4 x 4
                ('Conv2d_5', nn.Conv2d(64*8, 1, kernel_size=4, stride=1, padding=0)),
                ('Sigmoid', nn.Sigmoid()),
                ('Flatten', nn.Flatten())
                # Output: 1
            ])
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        output = self.main(inp)
        return output
