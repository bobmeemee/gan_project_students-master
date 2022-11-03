import torchvision.transforms

from utils import *
import argparse
from models import *


def generate_images(generator: nn.Module, config: dict, amount: int = 100,
                    denormalize: Denormalize = None) -> torch.Tensor:
    """
    Parameters:
        config (dict), should contain:
            latent_dim (int): the latent space dimension
            device (torch.device)
        amount (int): The amount of images to generate
        denormalize (Denormalize): The denormalize operation if needed, else None.
    """
    noise_vector = torch.randn(amount, config["latent_dim"], 1, 1, device=config["device"])
    images = generator(noise_vector)
    if denormalize != None:
        images = denormalize(images)
    return images


def save_images(images: torch.Tensor, config: dict):
    """
    Parameters:
        images (torch.Tensor): a tensor of shape (Batch_size, 3, Height, Width)
        config (dict), should contain:
            save_path (str)
            model (str): dcgan, wgan, wgan-gp, lsgan

    """
    transform = torchvision.transforms.ToPILImage()
    for i in range(images.shape[0]):
        pil_image = transform(images[i])
        pil_image.save(config['save_path'] + config["model"] + f"{i}.png")


if __name__ == "__main__":
    # parse arguments
    config = load_config("DCGAN")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--amount', type=int, help='How many images to generate.', default=100)
    parser.add_argument('--save_path', type=str, help='Path to save images.', default='images/')
    parser.add_argument('--discriminator', type=str, help='Path to discriminator weights',
                        default='test_saves/discr_DCGAN.pth')
    parser.add_argument('--generator', type=str, help='Path to generator weights', default='test_saves/gen_DCGAN.pth')
    parser.add_argument('--model', type=str, help='Type of model, choose from: wgan, wgan-gp, dcgan, lsgan',
                        default='dcgan')

    opt = parser.parse_args()
    args = vars(opt)
    for k, v in sorted(args.items()):
        if v is not None:
            config[k] = v
    config["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = config["device"]
    print(config)

    # ensure folders exists
    create_dir(config["save_path"])

    # Load models
    print("loading models...")
    generator, discriminator = load_gan(config)

    # generate images
    denormalize = Denormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    print("Generating images...")
    images = generate_images(generator, config, config["amount"], denormalize)
    print("Saving images...")
    save_images(images, config)
    print("Done!")
