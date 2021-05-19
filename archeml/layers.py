import keras
from keras.layers import LSTM, Conv1D
from keras.layers import BatchNormalization 


def _block_LSTM(filters, drop_rate, padding, inpR):
    'Returns LSTM block'    
    prev = inpR
    x_rnn = LSTM(filters, return_sequences=True, dropout=drop_rate, recurrent_dropout=drop_rate)(prev)
    NiN = Conv1D(filters, 1, padding = padding)(x_rnn)     
    res_out = BatchNormalization()(NiN)
    return res_out

def _block_BiGRU(filters, drop_rate, padding, inpR):
    'Returns LSTM residual block'    
    prev = inpR
    x_rnn = Bidirectional(GRU(filters, return_sequences=True, dropout=drop_rate, recurrent_dropout=drop_rate))(prev)
    NiN = Conv1D(filters, 1, padding = padding)(x_rnn)     
    res_out = BatchNormalization()(NiN)
    return res_out
    
