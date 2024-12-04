# Homework 4

## Original 4-Layer Model

![4-layer model](4-layer-model.png)

```
The 16 epoch achieves the best model, Test Loss: 0.1283, Test accuracy: 96.60%
```

## Modified 6-Layer Model

The 6-layer model required changes to pooling to be functional. When using 6 layers, the pooling layer shrunk the data to a size of 0 at the 5th layer and crashed the model.

```
RuntimeError: Given input size: (Nx1x1). Calculated output size: (Nx0x0). Output size is too small
```

My solution was to disable pooling for 2 of the layers.

### Growing hidden layer size

Continuing from layer 4, the number of channels was doubled in each layer (128 -> 256 -> 512)

The training was much slower than the smaller model, but reached its peak by epoch 13. Beyond that it began overfitting.

![6-layer model growing](6-layer-model-grow.png)

```
The 13 epoch achieves the best model, Test Loss: 0.1068, Test accuracy: 97.60%
```

There were other attempts at 6-layer models, but this is the one that I will use for the remainder of the modifications.

## Data Augmentation

### Random Affine

![Random Affine](random-affine.png)

```
The 18 epoch achieves the best model, Test Loss: 0.1074, Test accuracy: 97.40%
```

### Random Vertical Flip

![Random Vertical Flip](random-vertical-flip.png)

```
The 15 epoch achieves the best model, Test Loss: 0.2070, Test accuracy: 94.40%
```

## Other 6-layer experiments

These are the other, less successful 6-layer models.

### Shrinking hidden layer size

Starting at layer 4, the number of channels was cut in half in each layer (128 -> 64 -> 32)

The model was underfit on training data, but performed similarly to the 4-layer model on the test data.

![6-layer model shrinking](6-layer-model-shrink.png)

```
The 19 epoch achieves the best model, Test Loss: 0.2276, Test accuracy: 94.20%
```

### No size change

The additional 2 layers did not change the number of channels (128 -> 128 -> 128)

This model was overfit to the training data.

![6-layer model same size](6-layer-model-same-size.png)

```
The 20 epoch achieves the best model, Test Loss: 0.2486, Test accuracy: 92.70%
```

### "6"-layer model

This is a mock 6-layer model created by adding 4 more convolutional layers to the 4th layer. It performed almost the same as the best of the actual 6-layer models but I did not include it because it wasn't actually a 6-layer model.

![6-layer model growing](4-layer-model-4-extra.png)

The 16 epoch achieves the best model, Test Loss: 0.1077, Test accuracy: 97.50%
