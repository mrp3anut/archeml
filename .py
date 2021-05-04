from archml.helpers import result_metrics
import trainer as eqtt
from archml.models import mrp3anut
import pandas as pd



def test_number_of_layers():
    input_hdf5 = None
    input_csv = None
    output_name = None
    input_dimention = (6000, 3)
    cnn_blocks = 5
    lstm_blocks = 2
    padding = 'same'
    activation = 'relu'
    drop_rate = 0.1
    shuffle = True
    label_type = 'gaussian'
    normalization_mode = 'std'
    augmentation = True
    add_event_r = 0.6
    shift_event_r = 0.99
    add_noise_r = 0.3
    drop_channel_r = 0.5
    add_gap_r = 0.2
    scale_amplitude_r = None
    pre_emphasis = False
    loss_weights = [0.05, 0.40, 0.55]
    loss_types = ['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy']
    train_valid_test_split = [0.85, 0.05, 0.10]
    mode = 'generator'
    batch_size = 200
    epochs = 200
    monitor = 'val_loss'
    patience = 12
    multi_gpu = False
    number_of_gpus = 4
    gpuid = None
    gpu_limit = None
    use_multiprocessing = True
    model_select = mrp3anut

    args_dict = {
        "input_hdf5": input_hdf5,
        "input_csv": input_csv,
        "output_name": output_name,
        "input_dimention": input_dimention,
        "cnn_blocks": cnn_blocks,
        "lstm_blocks": lstm_blocks,
        "padding": padding,
        "activation": activation,
        "drop_rate": drop_rate,
        "shuffle": shuffle,
        "label_type": label_type,
        "normalization_mode": normalization_mode,
        "augmentation": augmentation,
        "add_event_r": add_event_r,
        "shift_event_r": shift_event_r,
        "add_noise_r": add_noise_r,
        "add_gap_r": add_gap_r,
        "drop_channel_r": drop_channel_r,
        "scale_amplitude_r": scale_amplitude_r,
        "pre_emphasis": pre_emphasis,
        "loss_weights": loss_weights,
        "loss_types": loss_types,
        "train_valid_test_split": train_valid_test_split,
        "mode": mode,
        "batch_size": batch_size,
        "epochs": epochs,
        "monitor": monitor,
        "patience": patience,
        "multi_gpu": multi_gpu,
        "number_of_gpus": number_of_gpus,
        "gpuid": gpuid,
        "gpu_limit": gpu_limit,
        "use_multiprocessing": use_multiprocessing,
        "model_select": model_select
    }
