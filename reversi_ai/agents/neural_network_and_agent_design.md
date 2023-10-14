# Agent

Uses a NN to predict where to place next piece

Use some kind of recursive algorithm to minmax? No, because it doesn't have a value head to calculate win probability. If it had, the policy head would have been used to pick the N "immediate" best moves so that the minmaxing could be sped up.

# Network design
Convolutional neural network, based on AlphaGo Zero. All convolutional networks have stride 1 and a padding such that the height and width is maintained. 2 residual blocks.

## Residual tower:

### Input Layer:
3x8x8 where the three layers are:
    white's pieces (1 or 0)
    black's pieces (1 or 0)
    color to play (constant 0 if white to play or 1 if black to play)

    conv with 3 filters, 3x3 kernel size

### Residual block:
Several residual blocks are used consecutively, with each one being

    conv with 32 filters, kernel size 3x3
    relu
    conv with 32 filters, kernel size 3x3
    skip connection from block input
    relu

At this stage the size of the input and output tensors are 32x8x8

### Policy head:
The policy head is what transforms the hidden layer result to a move.

    conv with 2 filters, kernel size 1x1
    relu
    fully connected layer from 2x8x8 (128) input features to and 8x8 (64) output

This output could then be put through a softmax if we want a non-greedy policy.
# Genetic mapping
The number of parameters in a 2d convolutional network is (n * m * l + 1) * k, where n and m are the kernel sizes, l is the number of filters and k is the number of feature map outputs. Each conv in the Residual block therefore have 9248.

Thus, with 2 Residual blocks the whole ResidualTowerPolicy will have (3 * 3 + 3 + 1) * 32 + 2 * 2 * 9248 + (1 * 1 * 32 + 1) * 2 + (128 + 1) * 64 = 45 730 parameters.

For comparison, the SimpleNN with 2 hidden layers that I've used previously has (65 + 1) * 32 + (32 + 1) * 32 + (32 + 1) * 64 = 5280 parameters.