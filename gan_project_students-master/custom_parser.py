import argparse


class CustomParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        ### TRAINING OPTIONS ###
        self.parser.add_argument('--num_epochs', type=int, default=20)
        self.parser.add_argument('--learning_rate', type=float, default=0.0002)
        self.parser.add_argument('--batch_size', type=int, default=128)
        self.parser.add_argument('--latent_dim', type=int, default=100)

        ### MISC ###
        self.parser.add_argument('--random_seed', type=int, default=1)
        self.parser.add_argument('--device', type=str, default="gpu")
        self.parser.add_argument('--name', type=str, default="GAN_01")
        self.parser.add_argument('--dataset_location', default="archive", type=str)
        self.parser.add_argument('--save_path', default="experiments/", type=str)
        self.parser.add_argument('--fid', action='store_true')
        self.parser.add_argument('--fid_amount', type=int, help='How many images to generate.', default=1000)
        self.parser.add_argument('--wandb', action='store_false')

    def parse(self, config: dict) -> dict:
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        for k, v in sorted(args.items()):
            if v is not None:
                config[k] = v

        return config
