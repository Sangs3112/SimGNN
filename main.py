from utils.config import Parser
from model.Trainer import Trainer
from utils.utils import set_seed, get_config, nice_printer, load_data

if __name__ == '__main__':
    args = Parser().parse()
    config = get_config(args)
    set_seed(config['seed'])

    train_data, test_data, norm_ged = load_data(args.dataset_path, args.dataset)
    nice_printer(config)
    trainer = Trainer(config, norm_ged)
    trainer.fit(train_data)
    trainer.score(train_data, test_data)
