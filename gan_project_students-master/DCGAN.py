import torch
from pytorch_fid.inception import InceptionV3

from models import *
from utils import *
from dataset import *
import torch.optim as optim
from torch.autograd import Variable
import wandb
from custom_parser import *
from fid_score import *

# Load configurations
config = CustomParser().parse({})
config["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = config["device"]
learning_rate = config["learning_rate"]
print(config)

# manual seeds
torch.manual_seed(config["random_seed"])
torch.cuda.manual_seed(config["random_seed"])

# Ensure save directories exists
create_dir(config["save_path"])
save_config(config)

# Load dataset
dataset = AnimeDataset(config)
denormalize = Denormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


# TIP 1: don't forget to apply the weights_init function from utils.py.
# TIP 2: don't forget to put the models on the correct device.
generator = Generator(config=config).to(device)
weights_init(generator)

discriminator = Discriminator().to(device)
weights_init(discriminator)


# Weights and biases
if config["wandb"]:
    wandb.login(key="284fe24200980a309d38f53c4cbfa6a628879b89")
    wandb.init(project="GAN_project", entity="ilib", name=config["name"] + "_train")
    wandb.config = {
        "learning_rate": config['learning_rate'],
        "epochs": config['num_epochs'],
        "batch_size": config['batch_size']
    }
    wandb.watch(generator, log='all')
    wandb.watch(discriminator, log='all')
    if config["fid"]:
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        model = InceptionV3([block_idx]).to(device)

# TODO: define loss function
adversarial_loss = nn.BCELoss()


G_optimizer = optim.Adam(generator.parameters(), lr=config["learning_rate"])
D_optimizer = optim.Adam(discriminator.parameters(), lr=config["learning_rate"])

# define all ones and all zeros tensors
real_target = Variable(torch.ones(config["batch_size"], 1).to(device))
fake_target = Variable(torch.zeros(config["batch_size"], 1).to(device))

# this fixed noise is for improvement visualisation only
fixed_noise = torch.randn(128, config["latent_dim"], 1, 1, device=device)

# print examples from dataset
for index, (images, _) in enumerate(dataset.train_loader):
    if config["wandb"]:
        wandb_images = wandb.Image(images, caption="Dataset examples")
        wandb.log({"examples": wandb_images})
    else:
        show_images(images, denormalize)
    break

for epoch in range(1, config["num_epochs"] + 1):
    D_loss_list, G_loss_list = [], []

    for index, (real_images, _) in enumerate(dataset.train_loader):
        D_optimizer.zero_grad()

        ## Discriminator real ##
        real_images = real_images.to(device)

        output = discriminator(real_images)
        D_real_loss = discriminator_loss(adversarial_loss, output, real_target)
        D_real_loss.backward()

        ## Discriminator fake ##
        noise_vector = torch.randn(config["batch_size"], config["latent_dim"], 1, 1, device=device)
        generated_image = generator.forward(noise_vector)

        output = discriminator.forward(generated_image.detach())
        D_fake_loss = discriminator_loss(adversarial_loss, output, fake_target)
        D_fake_loss.backward()

        D_optimizer.step()

        # Discriminator tot loss
        D_total_loss = D_real_loss + D_fake_loss
        D_loss_list.append(D_total_loss)

        if epoch % config["generator_epoch"] == 0:
            ## Train G on D's output ##
            G_optimizer.zero_grad()
            gen_output = discriminator.forward(generated_image)
            G_loss = generator_loss(adversarial_loss, gen_output, real_target)
            G_loss_list.append(G_loss)

            G_loss.backward()

            G_optimizer.step()


    discr_loss_mean = torch.mean(torch.FloatTensor(D_loss_list)).item()
    gen_loss_mean = torch.mean(torch.FloatTensor(G_loss_list)).item()
    print('Epoch: [%d/%d]: D_loss: %.3f, G_loss: %.3f' % (
        epoch, config["num_epochs"], discr_loss_mean,
        gen_loss_mean))

    if config["wandb"]:
        generated_image = denormalize(generator(fixed_noise))
        images = wandb.Image(generated_image, caption=f"epoch {epoch}")
        log = {
            "gen_loss": gen_loss_mean,
            "discr_loss": discr_loss_mean,
            "abs(gen_loss)": abs(gen_loss_mean),
            "abs(discr_loss)": abs(discr_loss_mean),
            "test images": images
        }
        if config["fid"]:
            fid_score = fid_score_generator(generator, model, config, denormalize=denormalize)
            log.update({
                "FID": fid_score
            })

        wandb.log(log)

    torch.save(generator.state_dict(), f"{config['save_path']}/generator_epoch_{epoch}.pth")
    torch.save(discriminator.state_dict(), f"{config['save_path']}/discriminator_epoch_{epoch}.pth")
