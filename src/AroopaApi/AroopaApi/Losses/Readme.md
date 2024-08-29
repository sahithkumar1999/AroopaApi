# CrossEntropyLoss Class

## Overview
The `CrossEntropyLoss` class provides methods to compute cross-entropy loss and softmax probabilities, commonly used in machine learning for evaluating classification models.

## Methods

### 1. ComputeCrossEntropyLoss 
#### Method Signature:
```csharp
public static double ComputeCrossEntropyLoss(
    NDArray input,
    NDArray target,
    double[] weight = null,
    int ignoreIndex = -100,
    string reduction = "mean",
    double labelSmoothing = 0.0)

```

#### Description:
Computes the cross-entropy loss between predicted logits or probabilities (`input`) and target class indices (`target`).

- `input`: 2D array representing predicted logits or probabilities (`batch_size x num_classes`).
- `target`: 1D array of target class indices (`batch_size`).
- `weight` (optional): 1D array of class weights for adjusting loss based on class importance.
- `ignoreIndex`: Index to ignore in loss computation (e.g., for padding).
- `reduction`: Type of reduction to apply to the loss ("mean" or "sum").
- `labelSmoothing`: Amount of label smoothing to apply (0.0 means no smoothing).

#### Example:

```csharp
using NumSharp;
using AroopaApi.Losses;

// Example usage
NDArray input = np.array(new double[,] { { 0.9, 0.1 }, { 0.4, 0.6 }, { 0.2, 0.8 } });
NDArray target = np.array(new int[] { 0, 1, 1 });

CrossEntropyLoss CrossEntropyLoss = new CrossEntropyLoss();
double loss = CrossEntropyLoss.ComputeCrossEntropyLoss(input, target);
Console.WriteLine($"Cross-Entropy Loss: {loss}");
```

### 2. Softmax
#### Method Signature:
```csharp
public static double[] Softmax(double[] logits)
```

#### Description:
Computes softmax probabilities from logits (scores).

- `logits`: Array of logits from which to compute probabilities.

#### Example:
```csharp
using AroopaApi.Losses;
double[] logits = { 2.0, 1.0, 0.1 };

CrossEntropyLoss CrossEntropyLoss = new CrossEntropyLoss();
double[] probabilities = CrossEntropyLoss.Softmax(logits);
Console.WriteLine($"Softmax Probabilities: [{string.Join(", ", probabilities)}]");
```