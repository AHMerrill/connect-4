USING THE TRAINED CONNECT-4 MODEL (.h5)
=======================================

YES — THIS IS A CONVOLUTIONAL NEURAL NETWORK (CNN)
-------------------------------------------------
The model is a residual CNN (AlphaZero-style):
- 2D convolutions over the 6x7 board
- Residual blocks
- Separate policy and value heads

It is NOT a transformer, RNN, or MLP.

-------------------------------------------------

IMPORTANT CONCEPT (READ THIS FIRST)
----------------------------------

The .h5 file is ONLY a neural network.

It does NOT:
- contain game rules
- contain Monte Carlo Tree Search (MCTS)
- detect forced wins or blocks by itself
- play Connect-4 on its own

All playing strength comes from HOW the model is wrapped in gameplay logic.

-------------------------------------------------

WHAT THE MODEL DOES
------------------

Input:
- Shape: (6, 7, 2)
- Channel 0: current player stones
- Channel 1: opponent stones
- Always from the CURRENT PLAYER perspective

Output:
- policy: length-7 vector (one value per column)
- value: scalar in range [-1, 1]

-------------------------------------------------

LOADING THE MODEL (REQUIRED)
---------------------------

Always load with compile=False.

Python:

import tensorflow as tf

model = tf.keras.models.load_model(
    "final_supervised_256f.h5",
    compile=False
)

-------------------------------------------------

MODE 1 — PURE POLICY INFERENCE (FAST, WEAK)
------------------------------------------

This is the simplest possible usage.
One forward pass, take argmax(policy).

Python:

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

Pros:
- Very fast
- Easy to integrate

Cons:
- Misses tactics
- Loses badly to MCTS
- Not strong by itself

-------------------------------------------------

MODE 2 — POLICY-GUIDED MCTS (STRONG, RECOMMENDED)
------------------------------------------------

This is how the model is meant to be used.

The CNN provides:
- policy → move priors
- value → leaf evaluation

MCTS provides:
- tactical lookahead
- win/block detection
- forced-line search

The CNN is called INSIDE MCTS at leaf nodes.

Pseudocode:

def mcts_evaluate(board):
    policy, value = model(board)
    return policy, value

During MCTS:
- policy initializes child priors
- value backs up the tree

Typical rollouts:
- 100  → decent
- 200  → strong
- 500+ → very strong

NOTE:
The .h5 file does NOT contain MCTS.
MCTS must be implemented separately.

-------------------------------------------------

ABOUT CONNECT-4
---------------

Connect-4 is a solved game.
Perfect play wins for the first player.

This means:
- 50% win rate vs strong MCTS is normal
- Improvement comes from better MCTS, not just a bigger CNN

-------------------------------------------------

SUMMARY
-------

- The .h5 is a CNN policy/value network
- It cannot play alone
- Pure policy is weak
- CNN + MCTS is strong
- Your results are expected and sane
