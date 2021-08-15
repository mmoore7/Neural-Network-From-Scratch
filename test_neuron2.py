import numpy as np
import neurons
from nnfs.datasets import spiral_data

X_test, y_test = spiral_data(samples=100, classes=3)

dense1 = neurons.LayerDense(2, 64)
activation1 = neurons.ActivationReLU()

dense2 = neurons.LayerDense(64, 3)
activation2 = neurons.ActivationSoftmax()

loss_activation = neurons.ActivationSoftmaxLossCategoricalCrossEntropy()

dense1.forward(X_test)

activation1.forward(dense1.output)

dense2.forward(activation1.output)

loss = loss_activation.forward(dense2.output, y_test)

predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions==y_test)

print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')
