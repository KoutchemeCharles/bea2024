from argparse import ArgumentParser
from src.nle.NLE import NLE
from src.repair.Repair import Repair
from src.utils.core import set_seed
from src.utils.files import read_config

def parse_args():
    parser = ArgumentParser(description="Running experiments")
    parser.add_argument("--config", required=True,
                        help="Path towards the configuration file")
    parser.add_argument('--test_run',
                        help="Whether to do a test run to ensure the pipeline works without issues",
                        action="store_true")

    return parser.parse_args()

def main():
    args = parse_args()
    config = read_config(args.config)
    set_seed(config.seed)

    if "repair" in config.task.name:
        experiment = Repair(config, test_run=args.test_run)
    elif config.task.name == "feedback":
        experiment = NLE(config, test_run=args.test_run)
    else:
        raise ValueError(f"Experiment {args.experiment} not implemented")
    
    experiment.run()


if __name__ == "__main__":
    main()