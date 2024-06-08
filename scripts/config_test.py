from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from argparse import ArgumentParser


# Running
def make_config(config_file: str, quiet=False):
    config = get_current_config()
    # parser = ArgumentParser(description="Fast imagenet training")
    # config.augment_argparse(parser)
    # config.collect_argparse_args(parser)
    config.collect_config_file(config_file)
    config.validate(mode="stderr")
    if not quiet:
        config.summary()


if __name__ == "__main__":
    parser = ArgumentParser(description="Fast imagenet training")
    parser.add_argument("config_file", type=str, help="Path to the config file")
    args = parser.parse_args()
    make_config(args.config_file)
