from pytorch_fid_master.src.pytorch_fid.fid_score import *
import torchvision.transforms as TR
from generate_images import *


def calculate_statistics_from_path(path, model, config, batch_size, dims,
                                   num_workers=1):
    path = pathlib.Path(path)
    files = []
    for root, _, files_in_dir in os.walk(path):
        for ext in IMAGE_EXTENSIONS:
            for file in files_in_dir:
                if file.endswith(ext):
                    files.append(path.joinpath(root, file))
    act = custom_get_activations(files, model, config, batch_size, dims, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_statistics_from_generator(generator: nn.Module, model: nn.Module, config, batch_size, dims, num_workers=1,
                                        denormalize: Denormalize = None):
    act = custom_get_activations(
        {
            "generator": generator,
            "denormalize": denormalize
        },
        model,
        config,
        batch_size,
        dims,
        num_workers
    )
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def custom_get_activations_batch(batch: torch.Tensor, model, start_idx: int, pred_arr: list, device: torch.device):
    batch = batch.to(device)

    with torch.no_grad():
        pred = model(batch)[0]

    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    pred = pred.squeeze(3).squeeze(2).cpu().numpy()

    pred_arr[start_idx:start_idx + pred.shape[0]] = pred

    start_idx = start_idx + pred.shape[0]

    return start_idx, pred_arr


def custom_get_activations(var, model, config, batch_size=50, dims=2048, num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- var       : List of image files paths OR a dictionary (containing 'generator', and 'denormalize')
    -- model       : Instance of inception model
    -- config       : The configurations (dict). Contains 'fid_amount' and 'device'.
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()
    start_idx = 0

    pred_arr = np.empty((config["fid_amount"], dims))
    if type(var) == list:
        if batch_size > len(var):
            print(('Warning: batch size is bigger than the data size. '
                   'Setting batch size to data size'))
            batch_size = len(var)

        dataset = ImagePathDataset(
            var,
            transforms=TR.Compose([
                TF.ToTensor(),
                TR.Resize([64, 64])
            ])
        )
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 drop_last=False,
                                                 num_workers=num_workers)

        for batch in dataloader:
            start_idx, pred_arr = custom_get_activations_batch(batch, model, start_idx, pred_arr, config["device"])
            if start_idx >= config["fid_amount"] - batch_size - 1:
                break
        return pred_arr
    elif type(var) == dict:
        if batch_size > config["fid_amount"]:
            print(('Warning: batch size is bigger than the data size. '
                   'Setting batch size to data size'))
            batch_size = config["fid_amount"]
        for i in range(int(config["fid_amount"] / batch_size)):
            images = generate_images(var["generator"], config, batch_size, var["denormalize"])
            start_idx, pred_arr = custom_get_activations_batch(images, model, start_idx, pred_arr, config["device"])
        return pred_arr
    else:
        raise TypeError("The input parameter 'var' should be a list of images, or a dict.")


def fid_score_generator(generator: torch.nn.Module, model: torch.nn.Module, config: dict, dims=2048, num_workers=1,
                        denormalize: Denormalize = None, log: bool = False) -> float:
    """
        Parameters:
            config (dict), should contain:
                latent_dim (int): the latent space dimension
                device (torch.device)
                dataset_location (str): The path towards the dataset.
            fid_amount (int): The fid_amount of images to generate
            denormalize (Denormalize): The denormalize operation if needed, else None.
            log (bool): print log statements.
        """

    if log:
        print("Start calculating statistics generator...")
    m1, s1 = calculate_statistics_from_generator(
        generator,
        model,
        config,
        config["batch_size"],
        dims,
        num_workers,
        denormalize
    )

    if log:
        print("Start calculating statistics dataset...")
    m2, s2 = calculate_statistics_from_path(
        config["dataset_location"],
        model,
        config,
        config["batch_size"],
        dims,
        num_workers
    )

    if log:
        print("Start calculating FID score...")
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    if log:
        print(f"The FID value is {fid_value}.")
    return fid_value


if __name__ == "__main__":
    # parse arguments
    config = load_config("DCGAN")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fid_amount', type=int, help='How many images to generate.', default=1000)
    parser.add_argument('--batch_size', type=int, help='How many images to generate.', default=100)
    parser.add_argument('--dataset_location', type=str, help='Path to dataset images',
                        default="C:/Users/Jens Duym/OneDrive - Universiteit Antwerpen/PHD/Datasets/anime")
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

    # set constants
    path = config["dataset_location"]
    dims = 2048

    if not os.path.exists(path):
        raise RuntimeError('Invalid path: %s' % path)

    # load models
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    generator, _ = load_gan(config)
    denormalize = Denormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    fid_score_generator(generator, model, config, dims, 1, denormalize, log=True)
