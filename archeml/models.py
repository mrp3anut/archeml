"""
models.py file contains modified versions of cred2 class found in EqTransformer code.
Cred2 is located on EQTransformer/EQTransformer/core/EqT_utils.py.
Original version of code can be found: 
https://github.com/smousavi05/EQTransformer/blob/906075feb8aaa61030ed6de1b02a6f749ef103a0/EQTransformer/core/EqT_utils.py#L2674
"""

from __future__ import division, print_function
import keras
from keras import backend as K
from keras.layers import add, Activation, LSTM, Conv1D
from keras.layers import MaxPooling1D, UpSampling1D, Cropping1D, SpatialDropout1D, Bidirectional, BatchNormalization 
from keras.models import Model
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
from obspy.signal.trigger import trigger_onset
from tensorflow.python.util import deprecation
from EQTransformer.core import EqT_utils
from EQTransformer.core.EqT_utils import _encoder, _transformer, _decoder, _block_CNN_1, _block_BiLSTM, SeqSelfAttention, _lr_schedule, f1
from .layers import _eqt_block_unidirectional_LSTM, _eqt_block_BiGRU

deprecation._PRINT_DEPRECATION_WARNINGS = False


"""    
EARTHML UPDATE: This is a modified version of cred2 from EqTransformer code. 
		In bilstm_closed model, BiLSTM block is removed.	
		
"""
class bilstm_closed():
    
    """ 
    
    Creates the bilstm_closed model
    
    Parameters
    ----------
    nb_filters: list
        The list of filter numbers. 
        
    kernel_size: list
        The size of the kernel to use in each convolutional layer.
        
    padding: str
        The padding to use in the convolutional layers.
    activationf: str
        Activation funciton type.
    endcoder_depth: int
        The number of layers in the encoder.
        
    decoder_depth: int
        The number of layers in the decoder.
    cnn_blocks: int
        The number of residual CNN blocks.
    BiLSTM_blocks: int=
        The number of Bidirectional LSTM blocks.
  
    drop_rate: float 
        Dropout rate.
    loss_weights: list
        Weights of the loss function for the detection, P picking, and S picking.       
                
    loss_types: list
        Types of the loss function for the detection, P picking, and S picking. 
    kernel_regularizer: str
        l1 norm regularizer.
    bias_regularizer: str
        l1 norm regularizer.
    multi_gpu: bool
        If use multiple GPUs for the training. 
    gpu_number: int
        The number of GPUs for the muli-GPU training. 
           
    Returns
    ----------
        The complied model: keras model
        
    """

    def __init__(self,
                 nb_filters=[8, 16, 16, 32, 32, 96, 96, 128],
                 kernel_size=[11, 9, 7, 7, 5, 5, 3, 3],
                 padding='same',
                 activationf='relu',
                 endcoder_depth=7,
                 decoder_depth=7,
                 cnn_blocks=5,
                 BiLSTM_blocks=3,
                 drop_rate=0.1,
                 loss_weights=[0.2, 0.3, 0.5],
                 loss_types=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],                                 
                 kernel_regularizer=keras.regularizers.l1(1e-4),
                 bias_regularizer=keras.regularizers.l1(1e-4),
                 multi_gpu=False, 
                 gpu_number=4, 
                 ):
        
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.padding = padding
        self.activationf = activationf
        self.endcoder_depth= endcoder_depth
        self.decoder_depth= decoder_depth
        self.cnn_blocks= cnn_blocks
        self.BiLSTM_blocks= BiLSTM_blocks     
        self.drop_rate= drop_rate
        self.loss_weights= loss_weights  
        self.loss_types = loss_types       
        self.kernel_regularizer = kernel_regularizer     
        self.bias_regularizer = bias_regularizer 
        self.multi_gpu = multi_gpu
        self.gpu_number = gpu_number

        
    def __call__(self, inp):

        x = inp
        x = _encoder(self.nb_filters, 
                    self.kernel_size, 
                    self.endcoder_depth, 
                    self.drop_rate, 
                    self.kernel_regularizer, 
                    self.bias_regularizer,
                    self.activationf, 
                    self.padding,
                    x)    
        
        for cb in range(self.cnn_blocks):
            x = _block_CNN_1(self.nb_filters[6], 3, self.drop_rate, self.activationf, self.padding, x)
            if cb > 2:
                x = _block_CNN_1(self.nb_filters[6], 2, self.drop_rate, self.activationf, self.padding, x)


    	# EARTHML UPDATE_start
        # for bb in range(self.BiLSTM_blocks):
        #    x = _block_BiLSTM(self.nb_filters[1], self.drop_rate, self.padding, x)
        # EARTHML UPDATE_end

            
        x, weightdD0 = _transformer(self.drop_rate, None, 'attentionD0', x)             
        encoded, weightdD = _transformer(self.drop_rate, None, 'attentionD', x)             
            
        decoder_D = _decoder([i for i in reversed(self.nb_filters)], 
                             [i for i in reversed(self.kernel_size)], 
                             self.decoder_depth, 
                             self.drop_rate, 
                             self.kernel_regularizer, 
                             self.bias_regularizer,
                             self.activationf, 
                             self.padding,                             
                             encoded)
        d = Conv1D(1, 11, padding = self.padding, activation='sigmoid', name='detector')(decoder_D)


        PLSTM = LSTM(self.nb_filters[1], return_sequences=True, dropout=self.drop_rate, recurrent_dropout=self.drop_rate)(encoded)
        norm_layerP, weightdP = SeqSelfAttention(return_attention=True,
                                                 attention_width= 3,
                                                 name='attentionP')(PLSTM)
        
        decoder_P = _decoder([i for i in reversed(self.nb_filters)], 
                            [i for i in reversed(self.kernel_size)], 
                            self.decoder_depth, 
                            self.drop_rate, 
                            self.kernel_regularizer, 
                            self.bias_regularizer,
                            self.activationf, 
                            self.padding,                            
                            norm_layerP)
        P = Conv1D(1, 11, padding = self.padding, activation='sigmoid', name='picker_P')(decoder_P)
        
        SLSTM = LSTM(self.nb_filters[1], return_sequences=True, dropout=self.drop_rate, recurrent_dropout=self.drop_rate)(encoded) 
        norm_layerS, weightdS = SeqSelfAttention(return_attention=True,
                                                 attention_width= 3,
                                                 name='attentionS')(SLSTM)
        
        
        decoder_S = _decoder([i for i in reversed(self.nb_filters)], 
                            [i for i in reversed(self.kernel_size)],
                            self.decoder_depth, 
                            self.drop_rate, 
                            self.kernel_regularizer, 
                            self.bias_regularizer,
                            self.activationf, 
                            self.padding,                            
                            norm_layerS) 
        
        S = Conv1D(1, 11, padding = self.padding, activation='sigmoid', name='picker_S')(decoder_S)
        

        if self.multi_gpu == True:
            parallel_model = Model(inputs=inp, outputs=[d, P, S])
            model = multi_gpu_model(parallel_model, gpus=self.gpu_number)
        else:
            model = Model(inputs=inp, outputs=[d, P, S])

        model.compile(loss=self.loss_types, loss_weights=self.loss_weights,    
            optimizer=Adam(lr=_lr_schedule(0)), metrics=[f1])

        return model

"""    
EARTHML UPDATE: This is a modified version of cred2 from EqTransformer code. 
		In genesisLSTM model, LSTM layers are removed from the pickers .	
		
"""
class genesisLSTM():
    
    """ 
    
    Creates the genesisLSTM model
    
    Parameters
    ----------
    nb_filters: list
        The list of filter numbers. 
        
    kernel_size: list
        The size of the kernel to use in each convolutional layer.
        
    padding: str
        The padding to use in the convolutional layers.
    activationf: str
        Activation funciton type.
    endcoder_depth: int
        The number of layers in the encoder.
        
    decoder_depth: int
        The number of layers in the decoder.
    cnn_blocks: int
        The number of residual CNN blocks.
    BiLSTM_blocks: int=
        The number of Bidirectional LSTM blocks.
  
    drop_rate: float 
        Dropout rate.
    loss_weights: list
        Weights of the loss function for the detection, P picking, and S picking.       
                
    loss_types: list
        Types of the loss function for the detection, P picking, and S picking. 
    kernel_regularizer: str
        l1 norm regularizer.
    bias_regularizer: str
        l1 norm regularizer.
    multi_gpu: bool
        If use multiple GPUs for the training. 
    gpu_number: int
        The number of GPUs for the muli-GPU training. 
           
    Returns
    ----------
        The complied model: keras model
        
    """

    def __init__(self,
                 nb_filters=[8, 16, 16, 32, 32, 96, 96, 128],
                 kernel_size=[11, 9, 7, 7, 5, 5, 3, 3],
                 padding='same',
                 activationf='relu',
                 endcoder_depth=7,
                 decoder_depth=7,
                 cnn_blocks=5,
                 BiLSTM_blocks=3,
                 drop_rate=0.1,
                 loss_weights=[0.2, 0.3, 0.5],
                 loss_types=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],                                 
                 kernel_regularizer=keras.regularizers.l1(1e-4),
                 bias_regularizer=keras.regularizers.l1(1e-4),
                 multi_gpu=False, 
                 gpu_number=4, 
                 ):
        
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.padding = padding
        self.activationf = activationf
        self.endcoder_depth= endcoder_depth
        self.decoder_depth= decoder_depth
        self.cnn_blocks= cnn_blocks
        self.BiLSTM_blocks= BiLSTM_blocks     
        self.drop_rate= drop_rate
        self.loss_weights= loss_weights  
        self.loss_types = loss_types       
        self.kernel_regularizer = kernel_regularizer     
        self.bias_regularizer = bias_regularizer 
        self.multi_gpu = multi_gpu
        self.gpu_number = gpu_number

        
    def __call__(self, inp):

        x = inp
        x = _encoder(self.nb_filters, 
                    self.kernel_size, 
                    self.endcoder_depth, 
                    self.drop_rate, 
                    self.kernel_regularizer, 
                    self.bias_regularizer,
                    self.activationf, 
                    self.padding,
                    x)    
        
        for cb in range(self.cnn_blocks):
            x = _block_CNN_1(self.nb_filters[6], 3, self.drop_rate, self.activationf, self.padding, x)
            if cb > 2:
                x = _block_CNN_1(self.nb_filters[6], 2, self.drop_rate, self.activationf, self.padding, x)

        for bb in range(self.BiLSTM_blocks):
            x = _block_BiLSTM(self.nb_filters[1], self.drop_rate, self.padding, x)

            
        x, weightdD0 = _transformer(self.drop_rate, None, 'attentionD0', x)             
        encoded, weightdD = _transformer(self.drop_rate, None, 'attentionD', x)             
            
        decoder_D = _decoder([i for i in reversed(self.nb_filters)], 
                             [i for i in reversed(self.kernel_size)], 
                             self.decoder_depth, 
                             self.drop_rate, 
                             self.kernel_regularizer, 
                             self.bias_regularizer,
                             self.activationf, 
                             self.padding,                             
                             encoded)
        d = Conv1D(1, 11, padding = self.padding, activation='sigmoid', name='detector')(decoder_D)

    	# EARTHML UPDATE_start
        # PLSTM = LSTM(self.nb_filters[1], return_sequences=True, dropout=self.drop_rate, recurrent_dropout=self.drop_rate)(encoded)
    	# EARTHML UPDATE_end


        norm_layerP, weightdP = SeqSelfAttention(return_attention=True,
                                                 attention_width= 3,
                                                 name='attentionP')(encoded)
        
        decoder_P = _decoder([i for i in reversed(self.nb_filters)], 
                            [i for i in reversed(self.kernel_size)], 
                            self.decoder_depth, 
                            self.drop_rate, 
                            self.kernel_regularizer, 
                            self.bias_regularizer,
                            self.activationf, 
                            self.padding,                            
                            norm_layerP)
        P = Conv1D(1, 11, padding = self.padding, activation='sigmoid', name='picker_P')(decoder_P)
        
    	# EARTHML UPDATE_start
        # SLSTM = LSTM(self.nb_filters[1], return_sequences=True, dropout=self.drop_rate, recurrent_dropout=self.drop_rate)(encoded) 
    	# EARTHML UPDATE_end

        norm_layerS, weightdS = SeqSelfAttention(return_attention=True,
                                                 attention_width= 3,
                                                 name='attentionS')(encoded)
        
        
        decoder_S = _decoder([i for i in reversed(self.nb_filters)], 
                            [i for i in reversed(self.kernel_size)],
                            self.decoder_depth, 
                            self.drop_rate, 
                            self.kernel_regularizer, 
                            self.bias_regularizer,
                            self.activationf, 
                            self.padding,                            
                            norm_layerS) 
        
        S = Conv1D(1, 11, padding = self.padding, activation='sigmoid', name='picker_S')(decoder_S)
        

        if self.multi_gpu == True:
            parallel_model = Model(inputs=inp, outputs=[d, P, S])
            model = multi_gpu_model(parallel_model, gpus=self.gpu_number)
        else:
            model = Model(inputs=inp, outputs=[d, P, S])

        model.compile(loss=self.loss_types, loss_weights=self.loss_weights,    
            optimizer=Adam(lr=_lr_schedule(0)), metrics=[f1])

        return model


"""    
EARTHML UPDATE: This is a modified version of cred2 from EqTransformer code. 
		In vanilla model, all of the LSTM and CNN blocks are removed, 
		leaving only the attention mechanisms and the sigmoid activation.	
		
"""
class vanilla():
    
    """ 
    
    Creates the vanilla model
    
    Parameters
    ----------
    nb_filters: list
        The list of filter numbers. 
        
    kernel_size: list
        The size of the kernel to use in each convolutional layer.
        
    padding: str
        The padding to use in the convolutional layers.
    activationf: str
        Activation funciton type.
    endcoder_depth: int
        The number of layers in the encoder.
        
    decoder_depth: int
        The number of layers in the decoder.
    cnn_blocks: int
        The number of residual CNN blocks.
    BiLSTM_blocks: int=
        The number of Bidirectional LSTM blocks.
  
    drop_rate: float 
        Dropout rate.
    loss_weights: list
        Weights of the loss function for the detection, P picking, and S picking.       
                
    loss_types: list
        Types of the loss function for the detection, P picking, and S picking. 
    kernel_regularizer: str
        l1 norm regularizer.
    bias_regularizer: str
        l1 norm regularizer.
    multi_gpu: bool
        If use multiple GPUs for the training. 
    gpu_number: int
        The number of GPUs for the muli-GPU training. 
           
    Returns
    ----------
        The complied model: keras model
        
    """

    def __init__(self,
                 nb_filters=[8, 16, 16, 32, 32, 96, 96, 128],
                 kernel_size=[11, 9, 7, 7, 5, 5, 3, 3],
                 padding='same',
                 activationf='relu',
                 endcoder_depth=7,
                 decoder_depth=7,
                 cnn_blocks=5,
                 BiLSTM_blocks=3,
                 drop_rate=0.1,
                 loss_weights=[0.2, 0.3, 0.5],
                 loss_types=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],                                 
                 kernel_regularizer=keras.regularizers.l1(1e-4),
                 bias_regularizer=keras.regularizers.l1(1e-4),
                 multi_gpu=False, 
                 gpu_number=4, 
                 ):
        
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.padding = padding
        self.activationf = activationf
        self.endcoder_depth= endcoder_depth
        self.decoder_depth= decoder_depth
        self.cnn_blocks= cnn_blocks
        self.BiLSTM_blocks= BiLSTM_blocks     
        self.drop_rate= drop_rate
        self.loss_weights= loss_weights  
        self.loss_types = loss_types       
        self.kernel_regularizer = kernel_regularizer     
        self.bias_regularizer = bias_regularizer 
        self.multi_gpu = multi_gpu
        self.gpu_number = gpu_number

        
    def __call__(self, inp):

        x = inp
        x = _encoder(self.nb_filters, 
                    self.kernel_size, 
                    self.endcoder_depth, 
                    self.drop_rate, 
                    self.kernel_regularizer, 
                    self.bias_regularizer,
                    self.activationf, 
                    self.padding,
                    x)    
    	
    	# EARTHML UPDATE_start        
        # for cb in range(self.cnn_blocks):
        #    x = _block_CNN_1(self.nb_filters[6], 3, self.drop_rate, self.activationf, self.padding, x)
        #    if cb > 2:
        #        x = _block_CNN_1(self.nb_filters[6], 2, self.drop_rate, self.activationf, self.padding, x)

        # for bb in range(self.BiLSTM_blocks):
        #    x = _block_BiLSTM(self.nb_filters[1], self.drop_rate, self.padding, x)
    	# EARTHML UPDATE_end


        x, weightdD0 = _transformer(self.drop_rate, None, 'attentionD0', x)             
        encoded, weightdD = _transformer(self.drop_rate, None, 'attentionD', x)             
            
        decoder_D = _decoder([i for i in reversed(self.nb_filters)], 
                             [i for i in reversed(self.kernel_size)], 
                             self.decoder_depth, 
                             self.drop_rate, 
                             self.kernel_regularizer, 
                             self.bias_regularizer,
                             self.activationf, 
                             self.padding,                             
                             encoded)
        d = Conv1D(1, 11, padding = self.padding, activation='sigmoid', name='detector')(decoder_D)

    	# EARTHML UPDATE_start
        # PLSTM = LSTM(self.nb_filters[1], return_sequences=True, dropout=self.drop_rate, recurrent_dropout=self.drop_rate)(encoded)
    	# EARTHML UPDATE_end

        norm_layerP, weightdP = SeqSelfAttention(return_attention=True,
                                                 attention_width= 3,
                                                 name='attentionP')(encoded)
        
        decoder_P = _decoder([i for i in reversed(self.nb_filters)], 
                            [i for i in reversed(self.kernel_size)], 
                            self.decoder_depth, 
                            self.drop_rate, 
                            self.kernel_regularizer, 
                            self.bias_regularizer,
                            self.activationf, 
                            self.padding,                            
                            norm_layerP)
        P = Conv1D(1, 11, padding = self.padding, activation='sigmoid', name='picker_P')(decoder_P)
    	
    	# EARTHML UPDATE_start
        # SLSTM = LSTM(self.nb_filters[1], return_sequences=True, dropout=self.drop_rate, recurrent_dropout=self.drop_rate)(encoded)  
    	# EARTHML UPDATE_end

        norm_layerS, weightdS = SeqSelfAttention(return_attention=True,
                                                 attention_width= 3,
                                                 name='attentionS')(encoded)
        
        
        decoder_S = _decoder([i for i in reversed(self.nb_filters)], 
                            [i for i in reversed(self.kernel_size)],
                            self.decoder_depth, 
                            self.drop_rate, 
                            self.kernel_regularizer, 
                            self.bias_regularizer,
                            self.activationf, 
                            self.padding,                            
                            norm_layerS) 
        
        S = Conv1D(1, 11, padding = self.padding, activation='sigmoid', name='picker_S')(decoder_S)
        

        if self.multi_gpu == True:
            parallel_model = Model(inputs=inp, outputs=[d, P, S])
            model = multi_gpu_model(parallel_model, gpus=self.gpu_number)
        else:
            model = Model(inputs=inp, outputs=[d, P, S])

        model.compile(loss=self.loss_types, loss_weights=self.loss_weights,    
            optimizer=Adam(lr=_lr_schedule(0)), metrics=[f1])

        return model

"""    
EARTHML UPDATE: This is a modified version of cred2 from EqTransformer code. 
		In gru model, all of the LSTMs are replaced with GRUs.		
"""
class gru():
    
    """ 
    
    Creates the gru model
    
    Parameters
    ----------
    nb_filters: list
        The list of filter numbers. 
        
    kernel_size: list
        The size of the kernel to use in each convolutional layer.
        
    padding: str
        The padding to use in the convolutional layers.
    activationf: str
        Activation funciton type.
    endcoder_depth: int
        The number of layers in the encoder.
        
    decoder_depth: int
        The number of layers in the decoder.
    cnn_blocks: int
        The number of residual CNN blocks.
    BiLSTM_blocks: int=
        The number of Bidirectional LSTM blocks.
  
    drop_rate: float 
        Dropout rate.
    loss_weights: list
        Weights of the loss function for the detection, P picking, and S picking.       
                
    loss_types: list
        Types of the loss function for the detection, P picking, and S picking. 
    kernel_regularizer: str
        l1 norm regularizer.
    bias_regularizer: str
        l1 norm regularizer.
    multi_gpu: bool
        If use multiple GPUs for the training. 
    gpu_number: int
        The number of GPUs for the muli-GPU training. 
           
    Returns
    ----------
        The complied model: keras model
        
    """

    def __init__(self,
                 nb_filters=[8, 16, 16, 32, 32, 96, 96, 128],
                 kernel_size=[11, 9, 7, 7, 5, 5, 3, 3],
                 padding='same',
                 activationf='relu',
                 endcoder_depth=7,
                 decoder_depth=7,
                 cnn_blocks=5,
                 BiLSTM_blocks=3,
                 drop_rate=0.1,
                 loss_weights=[0.2, 0.3, 0.5],
                 loss_types=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],                                 
                 kernel_regularizer=keras.regularizers.l1(1e-4),
                 bias_regularizer=keras.regularizers.l1(1e-4),
                 multi_gpu=False, 
                 gpu_number=4, 
                 ):
        
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.padding = padding
        self.activationf = activationf
        self.endcoder_depth= endcoder_depth
        self.decoder_depth= decoder_depth
        self.cnn_blocks= cnn_blocks
        self.BiLSTM_blocks= BiLSTM_blocks     
        self.drop_rate= drop_rate
        self.loss_weights= loss_weights  
        self.loss_types = loss_types       
        self.kernel_regularizer = kernel_regularizer     
        self.bias_regularizer = bias_regularizer 
        self.multi_gpu = multi_gpu
        self.gpu_number = gpu_number

        
    def __call__(self, inp):

        x = inp
        x = _encoder(self.nb_filters, 
                    self.kernel_size, 
                    self.endcoder_depth, 
                    self.drop_rate, 
                    self.kernel_regularizer, 
                    self.bias_regularizer,
                    self.activationf, 
                    self.padding,
                    x)    
        
        for cb in range(self.cnn_blocks):
            x = _block_CNN_1(self.nb_filters[6], 3, self.drop_rate, self.activationf, self.padding, x)
            if cb > 2:
                x = _block_CNN_1(self.nb_filters[6], 2, self.drop_rate, self.activationf, self.padding, x)




        for bb in range(self.BiLSTM_blocks):

        # EARTHML UPDATE_start
            x = _eqt_block_BiGRU(self.nb_filters[1], self.drop_rate, self.padding, x)
	# EARTHML UPDATE_end

            
        x, weightdD0 = _transformer(self.drop_rate, None, 'attentionD0', x)             
        encoded, weightdD = _transformer(self.drop_rate, None, 'attentionD', x)             
            
        decoder_D = _decoder([i for i in reversed(self.nb_filters)], 
                             [i for i in reversed(self.kernel_size)], 
                             self.decoder_depth, 
                             self.drop_rate, 
                             self.kernel_regularizer, 
                             self.bias_regularizer,
                             self.activationf, 
                             self.padding,                             
                             encoded)
        d = Conv1D(1, 11, padding = self.padding, activation='sigmoid', name='detector')(decoder_D)

	# EARTHML UPDATE_start
        PGRU = GRU(self.nb_filters[1], activation='tanh', return_sequences=True, dropout=self.drop_rate, recurrent_dropout=0)(encoded)
        # EARTHML UPDATE_end


        norm_layerP, weightdP = SeqSelfAttention(return_attention=True,
                                                 attention_width= 3,
                                                 name='attentionP')(PGRU)
        
        decoder_P = _decoder([i for i in reversed(self.nb_filters)], 
                            [i for i in reversed(self.kernel_size)], 
                            self.decoder_depth, 
                            self.drop_rate, 
                            self.kernel_regularizer, 
                            self.bias_regularizer,
                            self.activationf, 
                            self.padding,                            
                            norm_layerP)
        P = Conv1D(1, 11, padding = self.padding, activation='sigmoid', name='picker_P')(decoder_P)
        
        # EARTHML UPDATE_start
        SGRU = GRU(self.nb_filters[1],activation='tanh', return_sequences=True, dropout=self.drop_rate, recurrent_dropout=0)(encoded) 
        # EARTHML UPDATE_end
        
        norm_layerS, weightdS = SeqSelfAttention(return_attention=True,
                                                 attention_width= 3,
                                                 name='attentionS')(SGRU)
        
        
        decoder_S = _decoder([i for i in reversed(self.nb_filters)], 
                            [i for i in reversed(self.kernel_size)],
                            self.decoder_depth, 
                            self.drop_rate, 
                            self.kernel_regularizer, 
                            self.bias_regularizer,
                            self.activationf, 
                            self.padding,                            
                            norm_layerS) 
        
        S = Conv1D(1, 11, padding = self.padding, activation='sigmoid', name='picker_S')(decoder_S)
        

        if self.multi_gpu == True:
            parallel_model = Model(inputs=inp, outputs=[d, P, S])
            model = multi_gpu_model(parallel_model, gpus=self.gpu_number)
        else:
            model = Model(inputs=inp, outputs=[d, P, S])

        model.compile(loss=self.loss_types, loss_weights=self.loss_weights,    
            optimizer=Adam(lr=_lr_schedule(0)), metrics=[f1])

        return model


"""    
EARTHML UPDATE: This is a modified version of cred2 from EqTransformer code. 
		In lstm2 model, the LSTMs in the branches are doubled.		
"""
class lstm2():
    
    """ 
    
    Creates the lstm2 model
    
    Parameters
    ----------
    nb_filters: list
        The list of filter numbers. 
        
    kernel_size: list
        The size of the kernel to use in each convolutional layer.
        
    padding: str
        The padding to use in the convolutional layers.
    activationf: str
        Activation funciton type.
    endcoder_depth: int
        The number of layers in the encoder.
        
    decoder_depth: int
        The number of layers in the decoder.
    cnn_blocks: int
        The number of residual CNN blocks.
    BiLSTM_blocks: int=
        The number of Bidirectional LSTM blocks.
  
    drop_rate: float 
        Dropout rate.
    loss_weights: list
        Weights of the loss function for the detection, P picking, and S picking.       
                
    loss_types: list
        Types of the loss function for the detection, P picking, and S picking. 
    kernel_regularizer: str
        l1 norm regularizer.
    bias_regularizer: str
        l1 norm regularizer.
    multi_gpu: bool
        If use multiple GPUs for the training. 
    gpu_number: int
        The number of GPUs for the muli-GPU training. 
           
    Returns
    ----------
        The complied model: keras model
        
    """

    def __init__(self,
                 nb_filters=[8, 16, 16, 32, 32, 96, 96, 128],
                 kernel_size=[11, 9, 7, 7, 5, 5, 3, 3],
                 padding='same',
                 activationf='relu',
                 endcoder_depth=7,
                 decoder_depth=7,
                 cnn_blocks=5,
                 BiLSTM_blocks=3,
                 drop_rate=0.1,
                 loss_weights=[0.2, 0.3, 0.5],
                 loss_types=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],                                 
                 kernel_regularizer=keras.regularizers.l1(1e-4),
                 bias_regularizer=keras.regularizers.l1(1e-4),
                 multi_gpu=False, 
                 gpu_number=4, 
                 ):
        
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.padding = padding
        self.activationf = activationf
        self.endcoder_depth= endcoder_depth
        self.decoder_depth= decoder_depth
        self.cnn_blocks= cnn_blocks
        self.BiLSTM_blocks= BiLSTM_blocks     
        self.drop_rate= drop_rate
        self.loss_weights= loss_weights  
        self.loss_types = loss_types       
        self.kernel_regularizer = kernel_regularizer     
        self.bias_regularizer = bias_regularizer 
        self.multi_gpu = multi_gpu
        self.gpu_number = gpu_number

        
    def __call__(self, inp):

        x = inp
        x = _encoder(self.nb_filters, 
                    self.kernel_size, 
                    self.endcoder_depth, 
                    self.drop_rate, 
                    self.kernel_regularizer, 
                    self.bias_regularizer,
                    self.activationf, 
                    self.padding,
                    x)    
        
        for cb in range(self.cnn_blocks):
            x = _block_CNN_1(self.nb_filters[6], 3, self.drop_rate, self.activationf, self.padding, x)
            if cb > 2:
                x = _block_CNN_1(self.nb_filters[6], 2, self.drop_rate, self.activationf, self.padding, x)

        for bb in range(self.BiLSTM_blocks):
            x = _block_BiLSTM(self.nb_filters[1], self.drop_rate, self.padding, x)

            
        x, weightdD0 = _transformer(self.drop_rate, None, 'attentionD0', x)             
        encoded, weightdD = _transformer(self.drop_rate, None, 'attentionD', x)             
            
        decoder_D = _decoder([i for i in reversed(self.nb_filters)], 
                             [i for i in reversed(self.kernel_size)], 
                             self.decoder_depth, 
                             self.drop_rate, 
                             self.kernel_regularizer, 
                             self.bias_regularizer,
                             self.activationf, 
                             self.padding,                             
                             encoded)
        d = Conv1D(1, 11, padding = self.padding, activation='sigmoid', name='detector')(decoder_D)


        PLSTM = LSTM(self.nb_filters[1], return_sequences=True, dropout=self.drop_rate)(encoded)

	# EARTHML UPDATE_start   
        PLSTM2 = LSTM(self.nb_filters[1], return_sequences=True, dropout=self.drop_rate)(PLSTM)
        # EARTHML UPDATE_end

        norm_layerP, weightdP = SeqSelfAttention(return_attention=True,
                                                 attention_width= 3,
                                                 name='attentionP')(PLSTM2)
        
        decoder_P = _decoder([i for i in reversed(self.nb_filters)], 
                            [i for i in reversed(self.kernel_size)], 
                            self.decoder_depth, 
                            self.drop_rate, 
                            self.kernel_regularizer, 
                            self.bias_regularizer,
                            self.activationf, 
                            self.padding,                            
                            norm_layerP)
        P = Conv1D(1, 11, padding = self.padding, activation='sigmoid', name='picker_P')(decoder_P)
        
        SLSTM = LSTM(self.nb_filters[1], return_sequences=True, dropout=self.drop_rate)(encoded)

        # EARTHML UPDATE_start   
        SLSTM2 = LSTM(self.nb_filters[1], return_sequences=True, dropout=self.drop_rate)(SLSTM) 
        # EARTHML UPDATE_end

        norm_layerS, weightdS = SeqSelfAttention(return_attention=True,
                                                 attention_width= 3,
                                                 name='attentionS')(SLSTM2)
        
        
        decoder_S = _decoder([i for i in reversed(self.nb_filters)], 
                            [i for i in reversed(self.kernel_size)],
                            self.decoder_depth, 
                            self.drop_rate, 
                            self.kernel_regularizer, 
                            self.bias_regularizer,
                            self.activationf, 
                            self.padding,                            
                            norm_layerS) 
        
        S = Conv1D(1, 11, padding = self.padding, activation='sigmoid', name='picker_S')(decoder_S)
        

        if self.multi_gpu == True:
            parallel_model = Model(inputs=inp, outputs=[d, P, S])
            model = multi_gpu_model(parallel_model, gpus=self.gpu_number)
        else:
            model = Model(inputs=inp, outputs=[d, P, S])

        model.compile(loss=self.loss_types, loss_weights=self.loss_weights,    
            optimizer=Adam(lr=_lr_schedule(0)), metrics=[f1])

        return model

"""    
EARTHML UPDATE: This is a modified version of cred2 from EqTransformer code. 
		In bi_switch model, The Bidirectional LSTM block is removed and 
		LSTM (unidirectional) block is added in the place of BiLSTM block.	
"""
class bi_switch():
    """ 
    
    Creates the bi_switch model
    
    Parameters
    ----------
    nb_filters: list
        The list of filter numbers. 
        
    kernel_size: list
        The size of the kernel to use in each convolutional layer.
        
    padding: str
        The padding to use in the convolutional layers.
    activationf: str
        Activation funciton type.
    endcoder_depth: int
        The number of layers in the encoder.
        
    decoder_depth: int
        The number of layers in the decoder.
    cnn_blocks: int
        The number of residual CNN blocks.
    BiLSTM_blocks: int=
        The number of Bidirectional LSTM blocks.
  
    drop_rate: float 
        Dropout rate.
    loss_weights: list
        Weights of the loss function for the detection, P picking, and S picking.       
                
    loss_types: list
        Types of the loss function for the detection, P picking, and S picking. 
    kernel_regularizer: str
        l1 norm regularizer.
    bias_regularizer: str
        l1 norm regularizer.
    multi_gpu: bool
        If use multiple GPUs for the training. 
    gpu_number: int
        The number of GPUs for the muli-GPU training. 
           
    Returns
    ----------
        The complied model: keras model
        
    """

    def __init__(self,
                 nb_filters=[8, 16, 16, 32, 32, 96, 96, 128],
                 kernel_size=[11, 9, 7, 7, 5, 5, 3, 3],
                 padding='same',
                 activationf='relu',
                 endcoder_depth=7,
                 decoder_depth=7,
                 cnn_blocks=5,
                 BiLSTM_blocks=3,
                 drop_rate=0.1,
                 loss_weights=[0.2, 0.3, 0.5],
                 loss_types=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],                                 
                 kernel_regularizer=keras.regularizers.l1(1e-4),
                 bias_regularizer=keras.regularizers.l1(1e-4),
                 multi_gpu=False, 
                 gpu_number=4, 
                 ):
        
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.padding = padding
        self.activationf = activationf
        self.endcoder_depth= endcoder_depth
        self.decoder_depth= decoder_depth
        self.cnn_blocks= cnn_blocks
        self.BiLSTM_blocks= BiLSTM_blocks     
        self.drop_rate= drop_rate
        self.loss_weights= loss_weights  
        self.loss_types = loss_types       
        self.kernel_regularizer = kernel_regularizer     
        self.bias_regularizer = bias_regularizer 
        self.multi_gpu = multi_gpu
        self.gpu_number = gpu_number

        
    def __call__(self, inp):

        x = inp
        x = _encoder(self.nb_filters, 
                    self.kernel_size, 
                    self.endcoder_depth, 
                    self.drop_rate, 
                    self.kernel_regularizer, 
                    self.bias_regularizer,
                    self.activationf, 
                    self.padding,
                    x)    
        
        for cb in range(self.cnn_blocks):
            x = _block_CNN_1(self.nb_filters[6], 3, self.drop_rate, self.activationf, self.padding, x)
            if cb > 2:
                x = _block_CNN_1(self.nb_filters[6], 2, self.drop_rate, self.activationf, self.padding, x)
	
	# EARTHML UPDATE_start
        # for bb in range(self.BiLSTM_blocks):
        #    x = _block_BiLSTM(self.nb_filters[1], self.drop_rate, self.padding, x)


        for ls in range(self.BiLSTM_blocks):
            x = _eqt_block_unidirectional_LSTM(self.nb_filters[1], self.drop_rate, self.padding, x)
	# EARTHML UPDATE_end
            
        x, weightdD0 = _transformer(self.drop_rate, None, 'attentionD0', x)             
        encoded, weightdD = _transformer(self.drop_rate, None, 'attentionD', x)             
            
        decoder_D = _decoder([i for i in reversed(self.nb_filters)], 
                             [i for i in reversed(self.kernel_size)], 
                             self.decoder_depth, 
                             self.drop_rate, 
                             self.kernel_regularizer, 
                             self.bias_regularizer,
                             self.activationf, 
                             self.padding,                             
                             encoded)
        d = Conv1D(1, 11, padding = self.padding, activation='sigmoid', name='detector')(decoder_D)


        PLSTM = LSTM(self.nb_filters[1], return_sequences=True, dropout=self.drop_rate, recurrent_dropout=self.drop_rate)(encoded)
        norm_layerP, weightdP = SeqSelfAttention(return_attention=True,
                                                 attention_width= 3,
                                                 name='attentionP')(PLSTM)
        
        decoder_P = _decoder([i for i in reversed(self.nb_filters)], 
                            [i for i in reversed(self.kernel_size)], 
                            self.decoder_depth, 
                            self.drop_rate, 
                            self.kernel_regularizer, 
                            self.bias_regularizer,
                            self.activationf, 
                            self.padding,                            
                            norm_layerP)
        P = Conv1D(1, 11, padding = self.padding, activation='sigmoid', name='picker_P')(decoder_P)
        
        SLSTM = LSTM(self.nb_filters[1], return_sequences=True, dropout=self.drop_rate, recurrent_dropout=self.drop_rate)(encoded) 
        norm_layerS, weightdS = SeqSelfAttention(return_attention=True,
                                                 attention_width= 3,
                                                 name='attentionS')(SLSTM)
        
        
        decoder_S = _decoder([i for i in reversed(self.nb_filters)], 
                            [i for i in reversed(self.kernel_size)],
                            self.decoder_depth, 
                            self.drop_rate, 
                            self.kernel_regularizer, 
                            self.bias_regularizer,
                            self.activationf, 
                            self.padding,                            
                            norm_layerS) 
        
        S = Conv1D(1, 11, padding = self.padding, activation='sigmoid', name='picker_S')(decoder_S)
        

        if self.multi_gpu == True:
            parallel_model = Model(inputs=inp, outputs=[d, P, S])
            model = multi_gpu_model(parallel_model, gpus=self.gpu_number)
        else:
            model = Model(inputs=inp, outputs=[d, P, S])

        model.compile(loss=self.loss_types, loss_weights=self.loss_weights,    
            optimizer=Adam(lr=_lr_schedule(0)), metrics=[f1])

        return model

"""    
EARTHML UPDATE: This is a modified version of cred2 from EqTransformer code. 
		In bi_plus_lstm model, there is an addtional LSTM (unidirectional) 
		block right after Bidirectional LSTM block. 	
"""
class bi_plus_lstm():
    
    """ 
    
    Creates the bi_plus_lstm model
    
    Parameters
    ----------
    nb_filters: list
        The list of filter numbers. 
        
    kernel_size: list
        The size of the kernel to use in each convolutional layer.
        
    padding: str
        The padding to use in the convolutional layers.
    activationf: str
        Activation funciton type.
    endcoder_depth: int
        The number of layers in the encoder.
        
    decoder_depth: int
        The number of layers in the decoder.
    cnn_blocks: int
        The number of residual CNN blocks.
    BiLSTM_blocks: int=
        The number of Bidirectional LSTM blocks.
  
    drop_rate: float 
        Dropout rate.
    loss_weights: list
        Weights of the loss function for the detection, P picking, and S picking.       
                
    loss_types: list
        Types of the loss function for the detection, P picking, and S picking. 
    kernel_regularizer: str
        l1 norm regularizer.
    bias_regularizer: str
        l1 norm regularizer.
    multi_gpu: bool
        If use multiple GPUs for the training. 
    gpu_number: int
        The number of GPUs for the muli-GPU training. 
           
    Returns
    ----------
        The complied model: keras model
        
    """

    def __init__(self,
                 nb_filters=[8, 16, 16, 32, 32, 96, 96, 128],
                 kernel_size=[11, 9, 7, 7, 5, 5, 3, 3],
                 padding='same',
                 activationf='relu',
                 endcoder_depth=7,
                 decoder_depth=7,
                 cnn_blocks=5,
                 BiLSTM_blocks=3,
                 drop_rate=0.1,
                 loss_weights=[0.2, 0.3, 0.5],
                 loss_types=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],                                 
                 kernel_regularizer=keras.regularizers.l1(1e-4),
                 bias_regularizer=keras.regularizers.l1(1e-4),
                 multi_gpu=False, 
                 gpu_number=4, 
                 ):
        
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.padding = padding
        self.activationf = activationf
        self.endcoder_depth= endcoder_depth
        self.decoder_depth= decoder_depth
        self.cnn_blocks= cnn_blocks
        self.BiLSTM_blocks= BiLSTM_blocks     
        self.drop_rate= drop_rate
        self.loss_weights= loss_weights  
        self.loss_types = loss_types       
        self.kernel_regularizer = kernel_regularizer     
        self.bias_regularizer = bias_regularizer 
        self.multi_gpu = multi_gpu
        self.gpu_number = gpu_number

        
    def __call__(self, inp):

        x = inp
        x = _encoder(self.nb_filters, 
                    self.kernel_size, 
                    self.endcoder_depth, 
                    self.drop_rate, 
                    self.kernel_regularizer, 
                    self.bias_regularizer,
                    self.activationf, 
                    self.padding,
                    x)    
        
        for cb in range(self.cnn_blocks):
            x = _block_CNN_1(self.nb_filters[6], 3, self.drop_rate, self.activationf, self.padding, x)
            if cb > 2:
                x = _block_CNN_1(self.nb_filters[6], 2, self.drop_rate, self.activationf, self.padding, x)

        for bb in range(self.BiLSTM_blocks):
            x = _block_BiLSTM(self.nb_filters[1], self.drop_rate, self.padding, x)
        
	# EARTHML UPDATE_start    
        for ls in range(self.BiLSTM_blocks):
            x = _eqt_block_unidirectional_LSTM(self.nb_filters[1], self.drop_rate, self.padding, x)         
        # EARTHML UPDATE_end
	
        x, weightdD0 = _transformer(self.drop_rate, None, 'attentionD0', x)             
        encoded, weightdD = _transformer(self.drop_rate, None, 'attentionD', x)             
            
        decoder_D = _decoder([i for i in reversed(self.nb_filters)], 
                             [i for i in reversed(self.kernel_size)], 
                             self.decoder_depth, 
                             self.drop_rate, 
                             self.kernel_regularizer, 
                             self.bias_regularizer,
                             self.activationf, 
                             self.padding,                             
                             encoded)
        d = Conv1D(1, 11, padding = self.padding, activation='sigmoid', name='detector')(decoder_D)


        PLSTM = LSTM(self.nb_filters[1], return_sequences=True, dropout=self.drop_rate, recurrent_dropout=self.drop_rate)(encoded)
        norm_layerP, weightdP = SeqSelfAttention(return_attention=True,
                                                 attention_width= 3,
                                                 name='attentionP')(PLSTM)
        
        decoder_P = _decoder([i for i in reversed(self.nb_filters)], 
                            [i for i in reversed(self.kernel_size)], 
                            self.decoder_depth, 
                            self.drop_rate, 
                            self.kernel_regularizer, 
                            self.bias_regularizer,
                            self.activationf, 
                            self.padding,                            
                            norm_layerP)
        P = Conv1D(1, 11, padding = self.padding, activation='sigmoid', name='picker_P')(decoder_P)
        
        SLSTM = LSTM(self.nb_filters[1], return_sequences=True, dropout=self.drop_rate, recurrent_dropout=self.drop_rate)(encoded) 
        norm_layerS, weightdS = SeqSelfAttention(return_attention=True,
                                                 attention_width= 3,
                                                 name='attentionS')(SLSTM)
        
        
        decoder_S = _decoder([i for i in reversed(self.nb_filters)], 
                            [i for i in reversed(self.kernel_size)],
                            self.decoder_depth, 
                            self.drop_rate, 
                            self.kernel_regularizer, 
                            self.bias_regularizer,
                            self.activationf, 
                            self.padding,                            
                            norm_layerS) 
        
        S = Conv1D(1, 11, padding = self.padding, activation='sigmoid', name='picker_S')(decoder_S)
        

        if self.multi_gpu == True:
            parallel_model = Model(inputs=inp, outputs=[d, P, S])
            model = multi_gpu_model(parallel_model, gpus=self.gpu_number)
        else:
            model = Model(inputs=inp, outputs=[d, P, S])

        model.compile(loss=self.loss_types, loss_weights=self.loss_weights,    
            optimizer=Adam(lr=_lr_schedule(0)), metrics=[f1])

        return model