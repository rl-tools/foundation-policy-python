import h5py
import numpy as np
import json
import importlib_resources
import pathlib

class DenseLayer:
    def __init__(self, group):
        self.weights = group['weights']['parameters'][()]
        self.biases = group['biases']['parameters'][()]
        self.activation_function_name = group.attrs['activation_function']
        self.input_shape = [None, None, self.weights.shape[1]]
        self.output_shape = [None, None, self.weights.shape[0]]

    def description(self):
        return f"Dense({self.output_shape[2]})"

    def activation_function(self, x):
        if self.activation_function_name == "IDENTITY":
            return x
        elif self.activation_function_name == "RELU":
            return np.maximum(x, 0)
        elif self.activation_function_name == "SIGMOID":
            return 1 / (1 + np.exp(-x))
        elif self.activation_function_name == "TANH":
            return np.tanh(x)
        elif self.activation_function_name == "FAST_TANH":
            x = np.clip(x, -3.0, 3.0)
            x_squared = x * x
            return x * (27 + x_squared) / (27 + 9 * x_squared)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_function_name}")

    def evaluate(self, input_data):
        original_shape = input_data.shape
        leading_dims = original_shape[:-1]
        input_size = original_shape[-1]
        flattened_input = input_data.reshape(-1, input_size)
        output = self.evaluate_step(flattened_input)
        return output.reshape(*leading_dims, self.output_shape[-1])

    def evaluate_step(self, input_data):
        output = input_data @ self.weights.T + self.biases
        return self.activation_function(output)

class GRULayer:
    def __init__(self, group):
        self.weights_hidden = group['weights_hidden']['parameters'][()]
        self.weights_input = group['weights_input']['parameters'][()]
        self.hidden_dim = self.weights_input.shape[0] // 3
        self.input_shape = [None, None, self.weights_input.shape[1]]
        self.output_shape = [None, None, self.hidden_dim]
        self.biases_hidden = group['biases_hidden']['parameters'][()]
        self.biases_input = group['biases_input']['parameters'][()]
        self.initial_hidden_state = group['initial_hidden_state']['parameters'][()]
        self.state = None

    def description(self):
        return f"GRU({self.hidden_dim})"

    def reset(self):
        self.state = None

    def evaluate(self, input_data):
        if input_data.ndim == 2:
            return self.evaluate_step(input_data)
        elif input_data.ndim == 3:
            sequence_length, batch_size, _ = input_data.shape
            self.reset()
            outputs = [self.evaluate_step(input_data[i]) for i in range(sequence_length)]
            return np.stack(outputs, axis=0)
        else:
            raise ValueError("Input must be 2D or 3D")

    def evaluate_step(self, input_data):
        batch_size = input_data.shape[0]
        if self.state is None:
            self.state = np.tile(self.initial_hidden_state, (batch_size, 1))
        Wh = (self.weights_hidden @ self.state.T).T
        Wi = (self.weights_input @ input_data.T).T
        Wh_rz, Wh_n = Wh[:, :self.hidden_dim * 2], Wh[:, self.hidden_dim * 2:]
        Wi_rz, Wi_n = Wi[:, :self.hidden_dim * 2], Wi[:, self.hidden_dim * 2:]
        bh_rz, bh_n = self.biases_hidden[:self.hidden_dim * 2], self.biases_hidden[self.hidden_dim * 2:]
        bi_rz, bi_n = self.biases_input[:self.hidden_dim * 2], self.biases_input[self.hidden_dim * 2:]
        rz_pre = Wh_rz + Wi_rz + bh_rz + bi_rz
        rz = 1 / (1 + np.exp(-rz_pre))
        r, z = rz[:, :self.hidden_dim], rz[:, self.hidden_dim:]
        n_pre = r * (Wh_n + bh_n) + Wi_n + bi_n
        n = np.tanh(n_pre)
        self.state = z * self.state + (1 - z) * n
        return self.state

class SampleAndSquashLayer:
    def __init__(self, group):
        self.input_shape = [None, None, None]
        self.output_shape = [None, None, None]

    def description(self):
        return "SampleAndSquash"

    def evaluate(self, input_data):
        mean = input_data[..., :input_data.shape[-1] // 2]
        return np.tanh(mean)
    
    def evaluate_step(self, input_data):
        return self.evaluate(input_data)

class MLP:
    def __init__(self, group):
        self.input_layer = DenseLayer(group['input_layer'])
        self.hidden_layers = [DenseLayer(group[f'hidden_layer_{i}']) for i in range(group.attrs['num_layers'] - 2)]
        self.output_layer = DenseLayer(group['output_layer'])
        self.input_shape = self.input_layer.input_shape
        self.output_shape = self.output_layer.output_shape

    def description(self):
        hidden_desc = ", ".join(layer.description() for layer in self.hidden_layers)
        return f"MLP({self.input_layer.description()}, {hidden_desc}, {self.output_layer.description()})"

    def evaluate(self, input_data):
        current = self.input_layer.evaluate(input_data)
        for layer in self.hidden_layers:
            current = layer.evaluate(current)
        return self.output_layer.evaluate(current)
    
    def evaluate_step(self, input_data):
        return self.evaluate(input_data)

class Sequential:
    def __init__(self, group):
        layers_group = group['layers']
        self.layers = [layer_dispatch(layers_group[str(i)]) for i in range(len(layers_group))]
        self.input_shape = self.layers[0].input_shape
        self.output_shape = self.layers[-1].output_shape

    def description(self):
        return f"Sequential({', '.join(layer.description() for layer in self.layers)})"

    def reset(self):
        for layer in self.layers:
            if hasattr(layer, 'reset'):
                layer.reset()

    def evaluate(self, input_data):
        current = input_data
        for layer in self.layers:
            current = layer.evaluate(current)
        return current
    
    def evaluate_step(self, input_data):
        current = input_data
        for layer in self.layers:
            current = layer.evaluate_step(current)
        return current

def layer_dispatch(group):
    layer_type = group.attrs['type']
    if layer_type == 'dense':
        model = DenseLayer(group)
    elif layer_type == 'gru':
        model = GRULayer(group)
    elif layer_type == 'mlp':
        model = MLP(group)
    elif layer_type == 'sequential':
        model = Sequential(group)
    elif layer_type == 'sample_and_squash':
        model = SampleAndSquashLayer(group)
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")
    model.checkpoint_name = group.attrs.get('checkpoint_name', None)
    model.meta = json.loads(group.attrs['meta']) if 'meta' in group.attrs else None
    return model

def load(file_path):
    with h5py.File(file_path, 'r') as f:
        model = layer_dispatch(f['actor'])
        input_data = f['example']['input'][()]
        target_output = f['example']['output'][()]
        output = model.evaluate(input_data)
        diff = np.abs(output - target_output).mean()
        assert diff < 1e-6, "Output is not close enough to target output"
        return model


def QuadrotorPolicy():
    return load(importlib_resources.files("foundation_model").joinpath("blob", "checkpoint.h5"))