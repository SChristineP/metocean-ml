import torch
from metocean_ml.models import LSTM, FNN, LNN

# Test LSTM model
def test_lstm():
    input_size = 10
    hidden_size = 20
    output_size = 5
    num_layers = 2
    seq_len = 15
    batch_size = 4

    model = LSTM(input_size, hidden_size, output_size, num_layers)

    # Create a random input tensor with shape (batch_size, seq_len, input_size)
    x = torch.randn(batch_size, seq_len, input_size)

    # Forward pass
    output = model(x)

    # Check the output shape
    assert output.shape == (batch_size, output_size), f"Expected output shape: {(batch_size, output_size)}, got {output.shape}"

# Test LNN model
def test_lnn():
    input_size = 10
    output_size = 5
    dropout_rate = 0.2
    batch_size = 4

    model = LNN(input_size, output_size, dropout_rate)

    # Create a random input tensor
    x = torch.randn(batch_size, input_size)

    # Forward pass
    output = model(x)

    # Check the output shape
    assert output.shape == (batch_size, output_size), f"Expected output shape: {(batch_size, output_size)}, got {output.shape}"

# Test FNN model
def test_fnn():
    input_size = 10
    output_size = 5
    layers = [20, 30]
    dropout_rate = 0.2
    batch_normalization = True
    activ = "relu"
    batch_size = 4

    model = FNN(input_size, output_size, layers, dropout_rate, batch_normalization, activ)

    # Create a random input tensor
    x = torch.randn(batch_size, input_size)

    # Forward pass
    output = model(x)

    # Check the output shape
    assert output.shape == (batch_size, output_size), f"Expected output shape: {(batch_size, output_size)}, got {output.shape}"

# Test FNN with multiple dimensions
def test_fnn_with_multiple_dimensions():
    input_features = 10
    output_size = 5
    layers = [20, 30]
    dropout_rate = 0.2
    batch_normalization = True
    activ = "relu"
    batch_size = 4
    seq_len = 15
    input_size = input_features * seq_len

    model = FNN(input_size, output_size, layers, dropout_rate, batch_normalization, activ)

    # Create a random input tensor with 3 dimensions
    x = torch.randn(batch_size, seq_len, input_features)

    # Forward pass
    output = model(x)

    # Check the output shape
    assert output.shape == (batch_size, output_size), f"Expected output shape: {(batch_size, output_size)}, got {output.shape}"
