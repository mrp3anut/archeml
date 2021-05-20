import keras
from keras.layers import LSTM, Conv1D, BatchNormalization 

def _eqt_block_unidirectional_LSTM(filters, drop_rate, padding, inpR):
    
    """ 
    filters: int
        
    drop_rate: float 
        Dropout rate.
    
    padding: str
        The padding to use in the convolutional layers.
    
    inpR: 
        Input to the block 
    
    Returns:
        Unidirectional LSTM block
    """

    x_rnn = LSTM(filters, return_sequences=True, dropout=drop_rate, recurrent_dropout=drop_rate)(inpR)
    NiN = Conv1D(filters, 1, padding = padding)(x_rnn)     
    res_out = BatchNormalization()(NiN)
    return res_out

def _eqt_block_BiGRU(filters, drop_rate, padding, inpR):
    
    """
    filters: int
        
    drop_rate: float 
        Dropout rate.
    
    padding: str
        The padding to use in the convolutional layers.
    
    inpR: 
        Input to the block
    
    Returns:
        LSTM residual block
    """
    
    x_rnn = Bidirectional(GRU(filters, return_sequences=True, dropout=drop_rate, recurrent_dropout=drop_rate))(inpR)
    NiN = Conv1D(filters, 1, padding = padding)(x_rnn)     
    res_out = BatchNormalization()(NiN)
    return res_out