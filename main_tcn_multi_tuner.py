#  Copyright (c) 2022-2022 Andrew
#  Email: andrewlee1807@gmail.com
#
import os
import sys
import argparse
from ho_models import get_model, build_callbacks
from utils.data import Dataset, TimeSeriesGenerator
import yaml
import keras_tuner as kt
import tensorflow as tf

from utils.directory import create_file

# folder to load config file
CONFIG_PATH = "./"


# Get all arguments from command
def arg_parse(parser):
    parser.add_argument('--dataset_name', type=str, default='cnu', help='Dataset Name: household; cnu; spain; gyeonggi')
    parser.add_argument('--model_name', type=str, default='model1', help='Model Name: model1; model2; model3')
    parser.add_argument('--dataset_path', type=str, default='../dataset/', help='Dataset path')
    parser.add_argument('--config_path', type=str, help='Configuration file path')
    parser.add_argument('--search_mode', type=str, help='Searching mode')
    parser.add_argument('--output_length', type=int, default=1, help='Prediction Length')
    parser.add_argument('--max_trials', type=int, default=10, help='Max trials')
    parser.add_argument('--device', type=int, default=0, help='CUDA Device')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--write_log_file', type=bool, default=False,
                        help='Export to log file')  # if --write_log_file added, then we will export to log file
    return parser.parse_args()


def initialize_logging(file_name, pred_length):
    orig_stdout = sys.stdout
    if pred_length == 1:  # firstly execute, so we need to create a new file
        file_name = create_file(f'{file_name}')
    f = open(file_name, 'a')
    sys.stdout = f
    # Because of using a file to log, so we need to print the time to know when the program is running
    import datetime
    current_time = datetime.datetime.now().time()
    print("Running time - ", current_time)
    return f, orig_stdout


def close_logging(f, orig_stdout):
    print("====================================================================================================\n\n\n")
    sys.stdout = orig_stdout
    f.close()


def warming_up(args):
    """
    Configuring environment
    """
    print('Setting up environment...')
    # setup CUDA device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    print('Loading configuration file...')
    # read configuration
    with open(args.config_path, encoding='utf-8') as file:
        config = yaml.safe_load(file)
    config["output_length"] = args.output_length
    config["dataset_name"] = args.dataset_name
    config["max_trials"] = args.max_trials
    config["search_mode"] = args.search_mode
    config["tensorboard_log_dir"] = f'{args.output_dir}/tensorboard_log/{config["output_length"]}'
    print("Loaded configuration successfully ", args.config_path)

    print('Setting up output directory...')
    # initialize output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print("Output directory: ", args.output_dir)

    # initialize log file
    if args.write_log_file:
        print("Starting running background")
        print(
            f"To check output running, open file \"{os.path.join(args.output_dir, args.dataset_name)}_{config['output_length']} \"")
        file, orig_stdout = initialize_logging(f'{os.path.join(args.output_dir, args.dataset_name)}_training.log',
                                               config["output_length"])
        config["file"] = file
        config["orig_stdout"] = orig_stdout
    return config


def main():
    parser = argparse.ArgumentParser()
    args = arg_parse(parser)

    config = warming_up(args)

    # Load dataset
    dataset = Dataset(dataset_name=config["dataset_name"])
    # data = dataset.dataloader.export_a_single_sequence()
    data = dataset.dataloader.export_the_sequence(config["features"])

    # data = data[648:]

    # HOUSEHOLD dataset
    # dataloader = Dataset(dataset_name=args.dataset_name)
    # data = dataloader.dataloader.data_by_hour['Global_active_power']

    print("Building time series generator...")
    tsf = TimeSeriesGenerator(data=data,
                              config=config,
                              normalize_type=1,
                              shuffle=False)

    print("Building model...")
    # Get model (built and summary)
    model_builder = get_model(model_name=args.model_name,
                              config=config)
    max_trials = config["max_trials"]
    if config['search_mode'] == 'bayesian':
        exp_path = f"Tuning/Bayesian/{config['dataset_name']}" + "/" + str(config["dataset_name"]) + "/" + str(
            config["output_length"]) + "/"
    else:
        exp_path = f"Tuning/Hyperband/{config['dataset_name']}" + "/" + str(config["dataset_name"]) + "/" + str(
            config["output_length"]) + "/"
    tuning_path = exp_path + "/models"

    if os.path.isdir(tuning_path):
        import shutil
        shutil.rmtree(tuning_path)

    if config['search_mode'] == 'bayesian':
        tuner = kt.BayesianOptimization(
            model_builder,
            objective=kt.Objective("val_loss", direction="min"),
            max_trials=max_trials,
            seed=42,
            directory=tuning_path)
    else:
        tuner = kt.Hyperband(
            hypermodel=model_builder,
            objective=kt.Objective("val_loss", direction="min"),
            max_epochs=50,
            factor=3,
            seed=42,
            hyperband_iterations=1,
            distribution_strategy=tf.distribute.MirroredStrategy(),
            directory=tuning_path,
        )

    tuner.search_space_summary()

    tuner.search(tsf.data_train[0], tsf.data_train[1],
                 validation_data=tsf.data_valid,
                 callbacks=[tf.keras.callbacks.TensorBoard(exp_path + "/log")],
                 epochs=10)

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
    model_tuner = tuner.hypermodel.build(best_hps)

    # Train real model_searching
    print(f"""
            kernel_size {best_hps.get('kernel_size')},
            nb_filters: {best_hps.get('nb_filters')}, 
            dilations: {best_hps.get('dilations')}
            use_batch_norm: {best_hps.get('use_batch_norm')}
            use_skip_connections: {best_hps.get('use_skip_connections')}
    """)

    print('Train...')

    # callbacks
    callbacks = build_callbacks(tensorboard_log_dir=config["tensorboard_log_dir"])

    # Train model
    history = model_tuner.fit(x=tsf.data_train[0],  # [number_recoder, input_len, number_feature]
                              y=tsf.data_train[1],  # [number_recoder, output_len, number_feature]
                              validation_data=tsf.data_valid,
                              epochs=config["epochs"],
                              callbacks=[callbacks],
                              verbose=2,
                              batch_size=64,
                              use_multiprocessing=True)

    print("=============================================================")
    print("Minimum val mse:")
    print(min(history.history['val_mse']))
    print("Minimum training mse:")
    print(min(history.history['mse']))
    result = model_tuner.evaluate(tsf.data_test[0], tsf.data_test[1], batch_size=1,
                                  verbose=2,
                                  use_multiprocessing=True)
    print("Evaluation result: ")
    print(config['output_length'], result[1], result[2])

    result_file = f'{os.path.join(args.output_dir, args.dataset_name)}_evaluation_result.txt'
    file = open(result_file, 'a')
    file.write(f'{config["output_length"]} {result[1]} {result[2]}\n')
    file.close()

    if args.write_log_file:
        close_logging(config["file"], config["orig_stdout"])


if __name__ == '__main__':
    main()
