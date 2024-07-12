import argparse


def main(args: argparse.Namespace = None):
    print(f"{args.node_number}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node_number", type=str, required=True)
    args = parser.parse_args()
    main(args)
