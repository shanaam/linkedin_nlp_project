"""nn models for 

"""
import torch.nn as nn


class MLP(nn.Module):
    """
    A simple multilayer perceptron.
    """

    def __init__(self, in_dim, out_dim, hidden_dims=[], use_bias=True):
        """
    Constructs a MultiLayerPerceptron

    Args:
      in_dim: Integer
        dimensionality of input data (784)
      out_dim: Integer
        number of classes (10)
      hidden_dims: List
        containing the dimensions of the hidden layers,
        empty list corresponds to a linear model (in_dim, out_dim)

    Returns:
      Nothing
    """

        super(MLP, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        # If we have no hidden layer, just initialize a linear model (e.g. in logistic regression)
        if len(hidden_dims) == 0:
            layers = [nn.Linear(in_dim, out_dim, bias=use_bias)]
        else:
            # 'Actual' MLP with dimensions in_dim - num_hidden_layers*[hidden_dim] - out_dim
            layers = [nn.Linear(in_dim, hidden_dims[0], bias=use_bias), nn.ReLU()]

            # Loop until before the last layer
            for i, hidden_dim in enumerate(hidden_dims[:-1]):
                layers += [
                    nn.Linear(hidden_dim, hidden_dims[i + 1], bias=use_bias),
                    nn.ReLU(),
                ]

            # Add final layer to the number of classes
            layers += [nn.Linear(hidden_dims[-1], out_dim, bias=use_bias)]

        self.main = nn.Sequential(*layers)
        # NOTE: * unpacks a list! So it becomes nn.Sequential(layer[0], layer[1], ..., layer[n])

    def forward(self, x):
        """
    Defines the network structure and flow from input to output

    Args:
      x: Tensor
        Image to be processed by the network

    Returns:
      output: Tensor
        same dimension and shape as the input with probabilistic values in the range [0, 1]

    """
        # Flatten each images into a 'vector'
        output = self.main(x)
        # output = F.log_softmax(output, dim=1)
        return output
