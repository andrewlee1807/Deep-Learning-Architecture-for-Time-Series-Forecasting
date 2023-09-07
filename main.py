#  Copyright (c) 2022-2022 Andrew
#  Email: andrewlee1807@gmail.com
#
import os
import argparse
from models import get_model, build_callbacks
from utils.data import Dataset, TimeSeriesGenerator
from utils.logging import arg_parse, warming_up, close_logging


def main():
    parser = argparse.ArgumentParser()
    args = arg_parse(parser)

    config = warming_up(args)

    # Load dataset
    dataset = Dataset(dataset_name=config["dataset_name"])
    # data = dataset.dataloader.export_a_single_sequence()
    data = dataset.dataloader.export_the_sequence(config["features"])

    # data = data[648:]

    print("Building time series generator...")
    tsf = TimeSeriesGenerator(data=data,
                              config=config,
                              normalize_type=1,
                              shuffle=False)

    if "model" in config["dataset_name"]:  # delayNet model
        tsf.re_arrange_sequence(config)

    # tsf.normalize_data()

    print("Building model...")
    # Get model (built and summary)
    model = get_model(model_name=args.model_name,
                      config=config)

    # callbacks
    callbacks = build_callbacks(tensorboard_log_dir=config["tensorboard_log_dir"])

    # Train model
    history = model.fit(x=tsf.data_train[0],  # [number_recoder, input_len, number_feature]
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
    result = model.evaluate(tsf.data_test[0], tsf.data_test[1], batch_size=1,
                            verbose=2,
                            use_multiprocessing=True)
    print("Evaluation result: ")
    print(config['output_length'], result[1], result[2])

    result_file = f'{os.path.join(args.output_dir, args.dataset_name)}_evaluation_result.txt'
    file = open(result_file, 'a')
    file.write(f'{config["output_length"]},{result[1]},{result[2]}\n')
    file.close()

    if args.write_log_file:
        close_logging(config["file"], config["orig_stdout"])


if __name__ == '__main__':
    main()
