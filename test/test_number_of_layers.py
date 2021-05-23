
import os
from keras.models import load_model
from EQTransformer.core.EqT_utils import f1, SeqSelfAttention, FeedForward, LayerNormalization
from archeml import trainer as eqtt
from archeml.models import bilstm_closed


def extract_model_name():
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
    model_select = bilstm_closed

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

    model = eqtt._build_model( args_dict )

    return model


def h5_path_creator(model_name):

    file_name = str(model_name)
    path = os.path.join("ModelsAndSampleData", file_name)
    path = path  +'.h5'
    return path

def model_loader(path):


    model = load_model(path, custom_objects={'SeqSelfAttention': SeqSelfAttention,
                                        'FeedForward': FeedForward,
                                        'LayerNormalization': LayerNormalization,
                                        'f1': f1} )
    return len(model.layers)



def test_number_of_layers_():
    #model_1 corresponds to original eqt
    NUM_OF_LAYERS = {
        "model_1": 151,
        "bilstm_closed": 200,
        "mrp3anut_genesis": 400,
        "mrp3anut_vanilla": 500,
        "mrp3anut_gru": 150,
        "mrp3anut_lstm2": 250,
        "bclos": 350,
        "both_closed": 100,
        "both_open": 550,
        "model_2": 151}

    model_name = extract_model_name().name

    h5_path = h5_path_creator(model_name)

    x = model_loader(h5_path)

    y = [value for (key, value) in NUM_OF_LAYERS.items() if key == "{}".format( model_name )]

    assert x == y[0]