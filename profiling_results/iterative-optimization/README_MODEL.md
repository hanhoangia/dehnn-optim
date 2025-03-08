# Iterative Optimization Process

Our iterative optimization process has 3 stages: (1) Early Stopping -> (2) Architecture adjustments -> (3) Dynamic Learning Rate. The optimized model is the result of the last stage of this iterative optimization process, which involves refining the model through multiple cycles of optimization with a fixed random seed for weight initialization to control the effects of each optimization component.

The optimization technique of each stage is as follows:

- **Early Stopping**: A custom condition using 3 parameters: *patience*, *tolerance*, and *min-epochs*. Early Stopping is triggered when validation loss exceeds a *tolerance*-defined range around the average loss of the last *patience* epochs, but only after at least min-epochs have passed, preventing overfitting while ensuring a minimum number of training epochs.
- **Architecture Adjustments**: Grid Search is used to explore hyperparameter combinations and identify cost-accuracy trade-offs. The parameter grid consists of [2, 3, 4] for the number of layers and [8, 16, 32] for the number of dimensions.
- **Dynamic Learning Rate:** Cyclical Learning Rate (CLR) scheduler is employed to adjust the learning rate cyclically between a minimum and maximum value to accelerate convergence and avoid local minima.