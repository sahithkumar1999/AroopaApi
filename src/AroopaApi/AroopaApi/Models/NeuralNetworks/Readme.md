# Embedding



# BigramLanguageModel Class
## Overview

The `BigramLanguageModel` class implements a language model using token embeddings and softmax for prediction. It supports forward propagation, loss computation, and token sequence generation.

## Class Details

### Property:
```csharp
public Embedding TokenEmbeddingTable { get; private set; }
```
- Represents the embedding table for token representations, initialized with dimensions `(vocabSize, vocabSize)`.
## Constructor

### Method Signature:
```csharp
public BigramLanguageModel(int vocabSize)
```
- Initializes the language model with a specified vocabulary size (`vocabSize`).

## Methods
### 1. Forward

#### Method Signature:
```csharp
public (NDArray logits, NDArray targets) Forward(NDArray idx, NDArray targets = null)
```
- Performs forward propagation through the model to obtain logits.
- Returns a tuple containing:
  - `logits`: Model output logits (`batch_size x sequence_length x vocab_size`).
  - `targets`: Reshaped target indices if provided (`batch_size x sequence_length`).

##### Example:
```csharp
using NumSharp;

// Initialize the model
int vocabSize = 10000;
BigramLanguageModel model = new BigramLanguageModel(vocabSize);

// Example forward propagation
NDArray inputIndices = np.array(new int[,] { { 10, 20, 30 }, { 15, 25, 35 } });
var (logits, targets) = model.Forward(inputIndices);

```


