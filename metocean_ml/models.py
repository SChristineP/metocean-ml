import torch
from torch import nn


class LSTM(nn.Module):
    """
    A simple LSTM model for sequence prediction.

    Parameters
    ----------
    input_size : int
        The number of input features per timestep.
    hidden_size : int
        The number of hidden units in the LSTM.
    output_size : int
        The number of output features.
    num_layers : int
        The number of LSTM layers.
    dropout : float, optional
        The dropout rate applied to the LSTM layers, default is 0.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0):
        super().__init__()
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,bias=True,batch_first=True,dropout=dropout)
        self.out = nn.Linear(hidden_size,output_size)
    def forward(self,x):
        """
        Forward pass through the LSTM model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, seq_len, input_size).

        Returns
        -------
        torch.Tensor
            The output of the model with shape (batch_size, output_size).
        """
        x = torch.squeeze(self.lstm(x)[0][:,-1,:])
        return torch.squeeze(self.out(x))

class LNN(nn.Module):
    """
    A simple feedforward network with one linear layer and optional dropout.

    Parameters
    ----------
    input_size : int
        The number of input features.
    output_size : int
        The number of output features.
    dropout_rate : float, optional
        The dropout rate applied to the layer, default is 0.
    """
    def __init__(self, input_size,output_size,dropout_rate=0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_size,output_size),
                                 nn.Dropout(dropout_rate))
    def forward(self,x):
        """
        Forward pass through the linear network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            The output of the network.
        """
        if len(x.shape)>2:
            return self.net(torch.flatten(x,1))
        else: 
            return self.net(x)

class FNN(nn.Module):
    """
    A fully connected feedforward neural network with multiple layers.

    Parameters
    ----------
    input_size : int
        The number of input features.
    output_size : int
        The number of output features.
    layers : list of int
        The number of units in each hidden layer.
    dropout_rate : float, optional
        The dropout rate applied to the layers, default is 0.
    batch_normalization : bool, optional
        Whether to apply batch normalization, default is False.
    activ : str, optional
        The activation function to use ('relu', 'tanh', 'lrelu', or 'elu'), default is 'relu'.
    """
    def __init__(self, 
                 input_size,
                 output_size,
                 layers,
                 dropout_rate=0,
                 batch_normalization=False,
                 activ="relu"):
        super().__init__()
        modules = nn.ModuleList()

        modules.append(nn.Linear(input_size,layers[0]))

        if activ=="tanh": modules.append(nn.Tanh())
        if activ=="relu": modules.append(nn.ReLU())
        if activ=="lrelu": modules.append(nn.LeakyReLU())
        if activ=="elu": modules.append(nn.ELU())

        if batch_normalization: modules.append(nn.BatchNorm1d(layers[0]))

        if dropout_rate: modules.append(nn.Dropout(dropout_rate))

        for i in range(len(layers)-1):

            modules.append(nn.Linear(layers[i],layers[i+1]))

            if activ=="tanh": modules.append(nn.Tanh())
            if activ=="relu": modules.append(nn.ReLU())
            if activ=="lrelu": modules.append(nn.LeakyReLU())
            if activ=="elu": modules.append(nn.ELU())

            if batch_normalization: modules.append(nn.BatchNorm1d(layers[i+1]))

            if dropout_rate: modules.append(nn.Dropout(dropout_rate))

        modules.append(nn.Linear(layers[-1],output_size))
        self.net = nn.Sequential(*modules)

    def forward(self,x):
        """
        Forward pass through the feedforward network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            The output of the network.
        """
        if len(x.shape)>2:
            return self.net(torch.flatten(x,1))
        else: return self.net(x)
