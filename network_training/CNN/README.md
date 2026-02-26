# Using the Trained Connect-4 CNN Model

## Overview

YES — THIS IS A CONVOLUTIONAL NEURAL NETWORK (CNN)

The model is a residual CNN (AlphaZero-style):
- 2D convolutions over the 6×7 board
- Residual blocks
- Separate policy and value heads

It is **NOT** a transformer, RNN, or MLP.

## Important Concept (Read This First)

The `.h5` file is **ONLY a neural network**.

It does **NOT**:
- Contain game rules
- Contain Monte Carlo Tree Search (MCTS)
- Detect forced wins or blocks by itself
- Play Connect-4 on its own

**All playing strength comes from HOW the model is wrapped in gameplay logic.**

## What the Model Does

### Input
- **Shape:** `(6, 7, 2)`
- **Channel 0:** current player stones
- **Channel 1:** opponent stones
- Always from the **current player perspective**

### Output
- **Policy:** length-7 vector (one value per column)
- **Value:** scalar in range `[-1, 1]`

## Loading the Model (Required)

Always load with `compile=False`.

```python
import tensorflow as tf

model = tf.keras.models.load_model(
    "final_supervised_256f.h5",
    compile=False
)
```

## Mode 1: Pure Policy Inference (Fast, Weak)

This is the simplest possible usage. One forward pass, take `argmax(policy)`.

```python
import numpy as np

def policy_move(model, board):
    '''
    board: np.ndarray shape (6, 7, 2)
    returns: column int [0..6]
    '''
    x = board[None, ...]
    policy, _ = model.predict(x, verbose=0)
    policy = policy[0]

    # mask illegal columns
    illegal = board[:, :, 0].sum(axis=0) + board[:, :, 1].sum(axis=0) == 6
    policy[illegal] = -np.inf

    return int(np.argmax(policy))
```

**Pros:**
- Very fast
- Easy to integrate

**Cons:**
- Misses tactics
- Loses badly to MCTS
- Not strong by itself

## Mode 2: Policy-Guided MCTS (Strong, Recommended)

This is how the model is meant to be used.

The CNN provides:
- **Policy** → move priors
- **Value** → leaf evaluation

MCTS provides:
- Tactical lookahead
- Win/block detection
- Forced-line search

The CNN is called **inside MCTS at leaf nodes**.

### Pseudocode

```python
def mcts_evaluate(board):
    policy, value = model(board)
    return policy, value

# During MCTS:
# - policy initializes child priors
# - value backs up the tree
```

**Typical rollouts:**
- **100** → decent
- **200** → strong
- **500+** → very strong

**NOTE:** The `.h5` file does **NOT** contain MCTS. MCTS must be implemented separately.

## About Connect-4

Connect-4 is a **solved game**. Perfect play wins for the first player.

This means:
- 50% win rate vs strong MCTS is normal
- Improvement comes from better MCTS, not just a bigger CNN

## Summary

- The `.h5` is a CNN policy/value network
- It cannot play alone
- Pure policy is weak
- CNN + MCTS is strong
- Your results are expected and sane
