# Connect-4 AI: AlphaZero-Style Neural Networks for Game Play

## Table of Contents

1. [Data Generation](#1-data-generation-pipeline)
2. [Convolutional Neural Network Training](#2-cnn-training-and-model-selection)
3. [Transformer Network Training](#3-transformer-network-training)
4. [AWS Architecture and Setup](#4-aws-architecture-and-setup)
5. [Anvil Design and Gameplay](#5-anvil-app-design-and-gameplay)

---

## 1. Data Generation Pipeline

### 1.1 Purpose

The goal of the data generation pipeline is to create a large and diverse set of Connect-4 positions paired with high-quality supervision targets for training an AlphaZero-style policy and value network. The dataset is produced using Monte Carlo Tree Search (MCTS) to approximate strong play while intentionally encouraging broad state coverage.

### 1.2 Methodology

Game generation begins with a randomized opening phase designed to avoid collapse onto a narrow set of optimal openings. The number of random opening moves for each individual game played is sampled from a right-tailed distribution ranging from 4 to 14 with mode 6. This asymmetry improves coverage of vitally important mid and late-game states without neglecting early positions.

From each encountered position, MCTS is executed with a rollout budget of 2000 simulations. This budget reflects a tradeoff between label quality and dataset size. Higher simulation counts produce more accurate policy targets and smoother value estimates but reduce the total number of unique boards that can be generated under fixed compute constraints.

For each position, multiple statistics are recorded, including visit counts, aggregated scores, per-action Q values, and a scalar value target. The Q values represent the expected outcome of each action under search and provide a denser learning signal than terminal game outcomes alone. Positions are stored in a dictionary keyed by a canonical board representation so that repeated encounters accumulate additional statistics rather than creating duplicates.

### 1.3 Results

The final dataset contains approximately 700,000 unique board positions generated using MCTS with 2000 simulations per move. The exported dataset includes board tensors, policy distributions derived from visit counts, scalar value targets, and auxiliary search statistics such as visits, scores, and Q values. This dataset supports both standard AlphaZero-style supervised learning and deeper diagnostic analysis of search alignment.

**Resources:**
- Data generation code: [Google Drive](https://drive.google.com/file/d/1tq9ASdZPYUzMduKU36TS4skNxAv2MqJb/view?usp=sharing)
- Dataset (npz file): [GitHub Release v0.1-data](https://github.com/AHMerrill/connect-4/releases/tag/v0.1-data)

---

## 2. CNN Training and Model Selection

### 2.1 Purpose

The CNN training pipeline trains and evaluates residual convolutional neural networks on the generated dataset, with the goal of producing a policy and value function suitable for inference or for use as a prior inside a later MCTS. Emphasis is placed on data balancing, reproducible model selection, and efficient GPU execution.

### 2.2 Data Preparation

Because Connect-4 is a symmetric game when reflected across the center column, all boards are mirrored during preprocessing. This doubles the effective dataset size and reduces overfitting one-sided board positions.

Despite use of a distribution for random opening moves to attempt broad game-depth coverage, the dataset still exhibits strong imbalance across move depths, with early-game states appearing far more frequently than late-game states. This has the potential to create a problem in gameplay if the network overfits to early states and cannot generalize well to later stages of the game. To prevent over-learning these frequent regions, per-sample loss weights are computed based on move-depth bins. These weights are applied directly to the loss during training, ensuring that each depth region contributes more evenly to gradient updates without resampling or discarding data.

After weighting, the dataset is split into training and test sets using stratification over move-depth bins. This ensures that both sets share comparable depth distributions. The test split is used for model selection and early stopping rather than as a final performance benchmark.

### 2.3 Model Architecture

The network follows an AlphaZero-style residual convolutional architecture designed for small, fully observable board games.

#### 2.3.a Input Representation

Each board position is represented as a 6x7x2 tensor, where the two channels encode the current player's pieces and the opponent's pieces.

#### 2.3.b Shared Convolutional Trunk

The model begins with a 3x3 convolution that expands the input into a higher-dimensional feature space. This is followed by a stack of residual blocks. Each residual block consists of:

- A 3x3 convolution
- Batch normalization
- ReLU activation
- A second 3x3 convolution
- Batch normalization
- A skip connection that adds the block input to the block output
- ReLU activation

The skip connections allow the network to learn incremental refinements to features rather than entirely new transformations at each depth. This stabilizes optimization, mitigates vanishing gradients, and enables deeper networks without degradation in training performance.

Spatial resolution is preserved throughout the network rather than use of pooling; we felt that max-pooling would discard information that is relevant for move evaluation because the Connect-4 board is small and exact spatial relationships matter for tactics.

#### 2.3.c Policy and Value Heads

After the shared trunk, the network splits into two heads:

**Policy Head:**
- Produces a probability for each of the seven columns using a softmax activation
- Softmax is used because exactly one move must be selected, and it ensures the outputs form a valid probability distribution that sums to one
- This allows the model to express relative confidence across all possible moves based on what it has learned from various board positions

**Value Head:**
- Outputs a single scalar using a tanh activation, producing values between -1 and 1
- This range represents game outcomes from that particular board position, where -1 corresponds to likely loss, 0 to neutral or draw, and +1 to a likely win

The policy head guides which move to consider, while the value head estimates how good the position is overall, allowing the model to balance short-term tactics with long-term outcomes. Both are useful if gameplay will involve another MCTS; policy can choose a play directly, but value allows MCTS leaf evaluation and rollout replacement if simulations are used during actual gameplay.

For our gameplay, only policy was used, and no in-game MCTS is programmed into our app.

#### 2.3.d Model Capacity

The final architecture uses 10 residual blocks with 256 filters or feature maps per convolution, allowing the network to capture a wide range of board patterns and interactions. This resulted in approximately 35 million total parameters, with roughly 11.8 million trainable weights, and the remainder consisting of optimizer state and batch normalization parameters.

### 2.4 Model Selection Evaluation Philosophy and Final Training

Initial experiments compared networks with 8 versus 10 residual blocks and different learning rates while holding channel width fixed at 128 filters. The best-performing configuration used 10 residual blocks and a learning rate of 1e-3.

Validation loss and early stopping are used only during model comparison to select an architecture. The final model is then trained for a fixed number of epochs (informed but not directly chosen by early stopping numbers) without validation feedback and evaluated through gameplay against baseline random and MCTS opponents. This reflects our belief that playing strength, not loss values, ultimately defines performance in Connect-4.

A targeted capacity probe increased the number of convolutional filters from 128 to 256 while keeping the number of residual blocks fixed. This change reduced the best validation loss from 1.1818 to 1.1779, a small but consistent improvement. To determine whether this improvement reflected a meaningful difference in model behavior rather than noise, we evaluated how each model's value predictions aligned with outcome estimates produced by Monte Carlo Tree Search (MCTS) across a large set of positions and move depths.

In this context, calibration refers to how closely the network's value head agrees with what MCTS concludes after explicitly searching forward from a position. A well-calibrated value head produces estimates that are consistent with search-based evaluations, rather than merely fitting supervised training targets. The 256-filter model showed slightly stronger alignment with MCTS-derived outcomes overall, particularly in later and more difficult game states. Based on this combination of improved validation loss and better agreement with search-based evaluations, the wider model was selected as the final architecture.

The final supervised model was trained on the full mirrored dataset using 10 residual blocks, 256 filters, and 25 epochs. This model was exported as a .h5 file for downstream app use.

**Resources:**
- Model and training code: [Google Drive](https://drive.google.com/drive/folders/1qzKsT7cwXddIeu15dXuG_vK7q4rD_PnG?usp=sharing)

### 2.5 Implementation Notes

The model was trained on Google Colab using a GPU for faster performance. To speed up precision, most computations used lower-precision math, while critical parts stayed in full precision. Extra system optimizations suggested by OpenAI Codex were also used when possible to make training run faster overall.

### 2.6 CNN Model Architecture Table

| Layer (type) | Output Shape | Param # |
|--------------|--------------|---------|
| input_layer (InputLayer) | (None, 6, 7, 2) | 0 |
| conv2d (Conv2D) | (None, 6, 7, 256) | 4,608 |
| batch_normalization | (None, 6, 7, 256) | 1,024 |
| activation (Activation) | (None, 6, 7, 256) | 0 |
| *[10 Residual Blocks]* | (None, 6, 7, 256) | ~11.8M |
| conv2d_21 (Conv2D) - Policy | (None, 6, 7, 2) | 512 |
| conv2d_22 (Conv2D) - Value | (None, 6, 7, 1) | 256 |
| policy (Dense) | (None, 7) | 595 |
| value (Dense) | (None, 1) | 65 |

**Total params:** 35,458,838 (135.26 MB)
- **Trainable params:** 11,816,026 (45.07 MB)
- **Non-trainable params:** 10,758 (42.02 KB)
- **Optimizer params:** 23,632,054 (90.15 MB)

---

## 3. Transformer Network Training

### 3.1 Purpose

The Transformer training pipeline explores an alternative architecture to the CNN for learning Connect-4 policy and value functions. While CNNs leverage spatial locality through convolutions, Transformers use self-attention to learn relationships between any positions on the board. This approach treats the Connect-4 board as a sequence of patches (similar to Vision Transformers for images), enabling the model to capture long-range dependencies and complex positional patterns that may be relevant for strategic play.

### 3.2 Data Preparation

The same MCTS-generated dataset is used for Transformer training, with identical preprocessing steps:

**Board Representation:**
- Each board position is reshaped from (6, 7, 2) to (42, 2)
- The 42 patches correspond to the 42 cells of the board (6 rows x 7 columns)
- Each patch has 2 features: current player's stone presence and opponent's stone presence

**Data Augmentation:**
- Horizontal mirroring doubles the dataset from 703,111 to 1,406,222 samples
- Mirroring is applied by reversing columns in both the board representation and the policy targets
- Value targets remain unchanged since the expected outcome is symmetric

**Train/Validation Split:**
- 90% training (1,265,599 samples)
- 10% validation (140,623 samples)
- Random shuffle with fixed seed for reproducibility

### 3.3 Model Architecture

The Transformer follows a Vision Transformer (ViT) design adapted for the Connect-4 board game domain.

#### 3.3.a Input Processing

Each cell of the 6x7 board is treated as a "patch" with 2 features (current player stones, opponent stones). The input shape is (42, 2), where 42 = 6 rows x 7 columns. A dense layer projects each patch from 2 features to the hidden dimension (128).

#### 3.3.b Positional Embeddings

Since Transformers have no inherent notion of position, learned positional embeddings are added to each patch. These embeddings allow the model to distinguish between different board positions and learn position-dependent strategies (e.g., center control is often advantageous in Connect-4).

#### 3.3.c Class Token

Following the standard ViT approach, a learnable class token is prepended to the sequence. This token aggregates global information through self-attention and is used by the value head to predict game outcomes. After adding the class token, the sequence length becomes 43 (1 class token + 42 board patches).

#### 3.3.d Transformer Encoder Blocks

The model uses 6 stacked Transformer encoder blocks. Each block consists of:

**Multi-Head Self-Attention:**
- 8 attention heads
- Key dimension: 16 (hidden_dim / num_heads)
- Allows the model to attend to relevant positions across the entire board
- Pre-layer normalization for training stability

**Feed-Forward Network (MLP):**
- Hidden expansion to 256 dimensions
- GELU activation function
- Projects back to 128 dimensions
- Dropout (0.1) for regularization

**Residual Connections:**
- Skip connections around both attention and MLP sub-layers
- Enables gradient flow and allows learning incremental refinements

#### 3.3.e Policy Head

The policy head produces move probabilities for each of the 7 columns:

1. Extract the 42 patch tokens (excluding the class token)
2. Reshape to (6, 7, hidden_dim) to recover spatial structure
3. Average pool over rows to get column features: (7, hidden_dim)
4. Dense layer (64 units, ReLU) for column-wise processing
5. Dense layer (1 unit) per column to produce logits
6. Softmax activation for probability distribution

This design aggregates information from all rows within each column, allowing the model to assess column quality based on the full vertical context.

#### 3.3.f Value Head

The value head predicts the expected game outcome:

1. Extract the class token (index 0), which has aggregated global board information
2. Dense layer (128 units, ReLU)
3. Dropout (0.1)
4. Dense layer (64 units, ReLU)
5. Dense layer (1 unit) with tanh activation

The output is a scalar in [-1, +1] representing the expected outcome from the current player's perspective.

### 3.4 Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Hidden Dimension | 128 |
| Transformer Layers | 6 |
| Attention Heads | 8 |
| MLP Dimension | 256 |
| Dropout Rate | 0.1 |
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-4 |
| Batch Size | 512 |
| Max Epochs | 50 |

**Loss Functions:**
- Policy: Categorical cross-entropy
- Value: Mean squared error
- Combined loss: Equal weighting (1.0 each)

**Callbacks:**
- Early stopping on validation policy accuracy (patience: 10 epochs)
- Learning rate reduction on validation loss plateau (factor: 0.5, patience: 5 epochs)
- Best weights restoration

### 3.5 Training Results

| Metric | Value |
|--------|-------|
| Total Loss | 1.4064 |
| Policy Loss | 1.3956 |
| Value Loss | 0.0108 |
| Policy Accuracy | 77.29% |
| Policy Top-2 Accuracy | 92.39% |
| Value MAE | 0.0617 |

**Interpretation:**
- The model correctly predicts the MCTS-preferred move 77% of the time
- The correct move is in the model's top 2 choices 92% of the time
- Value predictions are highly accurate with mean absolute error of only 0.06

**Empty Board Prediction (Sanity Check):**
```
Policy: [0.085, 0.101, 0.171, 0.297, 0.162, 0.099, 0.084]
Value: 0.151
Best column: 3 (center)
```

The model correctly identifies the center column as the strongest opening move, consistent with Connect-4 strategy.

### 3.6 Model Capacity

The Transformer model has substantially fewer trainable parameters than the final CNN while maintaining competitive policy accuracy.

| Component | Parameters |
|-----------|------------|
| Patch Projection | 384 |
| Positional Embeddings | 5,376 |
| Class Token | 128 |
| 6 Transformer Blocks | 794,880 |
| Policy Head | 8,321 |
| Value Head | 24,833 |
| **Total** | **834,178** |

Compared with the final CNN (~11.8M trainable parameters), the Transformer uses about 14x fewer trainable weights.

### 3.7 Comparison: CNN vs Transformer

| Aspect | CNN | Transformer |
|--------|-----|-------------|
| Trainable Parameters | 11.8M | 834,178 |
| Inductive Bias | Local spatial patterns | Global attention |
| Policy Accuracy | ~75-78%* | 77.29% |
| Inference Speed | Faster | Slower |
| Strength | Proven architecture | Parameter efficient |

*CNN accuracy varies based on exact configuration and training details.

Both architectures achieve strong performance. In this project, the CNN provides high-capacity local pattern modeling, while the Transformer provides competitive policy quality with much lower parameter count.

### 3.8 Implementation Notes

- Trained on Google Colab with GPU acceleration (T4)
- TensorFlow 2.19.0 with Keras
- Custom Keras layers for positional embeddings, class token, and transformer blocks
- Model weights saved separately from architecture for deployment flexibility

---

## 4. AWS Architecture and Setup

### 4.1 Purpose

The AWS backend provides a scalable inference endpoint for the trained models, enabling the Anvil web application to request move predictions without requiring client-side model loading. This separation of concerns allows for a lightweight frontend while maintaining the computational requirements on dedicated server infrastructure.

### 4.2 Architecture Overview

```
┌─────────────────┐     HTTPS      ┌─────────────────┐
│   Anvil App     │ ──────────────>│   AWS Backend   │
│   (Frontend)    │                │   (Inference)   │
└─────────────────┘                └─────────────────┘
                                           │
                                           v
                                   ┌───────────────┐
                                   │ TensorFlow    │
                                   │ Model Weights │
                                   └───────────────┘
```

### 4.3 Backend Implementation

The AWS backend code includes all necessary components for model inference:

**Custom Keras Layers:**
- `PositionalEmbedding`: Learned position encodings for board patches
- `ClassToken`: Prepends learnable class token for value head
- `TransformerBlock`: Self-attention and feed-forward layers

**Model Building Function:**
```python
def build_connect4_transformer(
    num_rows=6, num_cols=7, patch_features=2,
    hidden_dim=128, num_layers=6, num_heads=8,
    mlp_dim=256, dropout_rate=0.1
):
    # Builds the full Transformer architecture
    # Returns: Keras Model with policy and value outputs
```

**Prediction Function:**
```python
def transformer_predict_move(board_6x7x2):
    """
    Predict best move using the transformer.

    Args:
        board_6x7x2: numpy array shape (6, 7, 2)
                     channel 0 = current player stones
                     channel 1 = opponent stones

    Returns:
        best_column: int 0-6 (the column to play)
    """
    board_flat = board_6x7x2.reshape(1, 42, 2).astype(np.float32)
    policy, value = transformer_model.predict(board_flat, verbose=0)
    return int(np.argmax(policy[0]))
```

### 4.4 Deployment Requirements

**Dependencies:**
- Python 3.8+
- TensorFlow 2.x
- NumPy

**Model Files:**
- `transformer.weights.h5` - Trained Transformer weights
- `cnn.h5` - Trained CNN model (optional, for CNN endpoint)

### 4.5 API Interface

The backend exposes `process_move(board_data)` via Anvil Uplink.

**Input (`board_data` dictionary):**
- `board`: 6x7 integer board (`1=plus`, `-1=minus`, `0=empty`)
- `current_player`: `'plus'`, `'minus'`, `1`, `2`, or `-1`
- `model_type`: `'CNN'` or `'Transformer'`

**Backend behavior:**
- If `current_player` is minus, board values are multiplied by `-1` so models always evaluate from the current-player perspective.
- Runs selected model if loaded; otherwise uses a center-preferring fallback policy.
- Applies legality checks and fallback correction if needed.

**Output:**
- `status`: `'success'` or `'error'`
- `recommended_move`: integer column index `0-6`
- `model_used`: model name string
- `symmetry_applied`: boolean
- `using_real_model`: boolean

### 4.6 Scaling Considerations

- **Stateless Design:** Each prediction request is independent, enabling horizontal scaling
- **Model Caching:** Models are loaded once at startup and kept in memory
- **Batch Inference:** Multiple requests can be batched for improved throughput
- **Cold Start:** Initial request may have higher latency due to model loading

---

## 5. Anvil App Design and Gameplay

### 5.1 Purpose

The Anvil application provides an interactive web interface for users to play Connect-4 against the trained AI models. The app handles game state management, board visualization, user input, and communication with the AWS backend for AI move generation.

### 5.2 Game Engine

The game engine implements standard Connect-4 rules:

**Board Representation:**
- 6 rows x 7 columns grid
- Player 1 represented as +1, Player 2 as -1, empty cells as 0
- Gravity-based piece placement (pieces fall to lowest available row)

**Core Functions:**

```python
def update_board(board, color, column):
    """Place a piece in the specified column."""
    # Finds lowest empty row in column and places piece

def check_for_win(board, col):
    """Check if the last move resulted in a win."""
    # Checks horizontal, vertical, and both diagonal directions
    # Returns 'plus', 'minus', or 'nobody'

def find_legal(board):
    """Return list of columns that are not full."""
    # Returns columns where top row is empty

def is_draw(board):
    """Check if the board is full (draw)."""
    # Returns True if no legal moves remain
```

**Board-to-Neural-Network Conversion:**
```python
def board_to_nn_input(board, player):
    """
    Convert board to neural network input format.
    Always from the perspective of the current player.

    Returns: (6, 7, 2) array
        Channel 0: Current player's pieces
        Channel 1: Opponent's pieces
    """
```

### 5.3 User Interface

**Board Visualization:**
- Graphical 6x7 board with colored chips (Burnt Orange / White)
- Hover-row chip preview for human turns
- Visual highlighting of the last move
- Win/draw detection and announcement

**Game Controls:**
- Model selection (CNN or Transformer)
- Side selection (Player 1 Burnt Orange or Player 2 White)
- Click-to-drop interaction by column
- New game, rematch, and quit buttons
- Setup-page footer links: How to Play, Model Report, Credits

### 5.4 AI Integration

AI turns are processed by calling backend `process_move`, which:

- Prepares board perspective for current player (symmetry flip for minus player)
- Runs selected model wrapper (`get_cnn_move` or `get_transformer_move`)
- Masks/filters invalid moves via legality checks
- Falls back to a deterministic center-priority move policy when model is unavailable

This deployment path uses direct model inference for low-latency gameplay (no in-game MCTS loop).

### 5.5 Game Modes

**Human vs AI:**
1. User selects opponent model (Transformer or CNN)
2. User chooses who plays first
3. Turn-based play with board visualization after each move
4. Game ends on win or draw with result announcement

**AI vs AI:**
- Watch Transformer play against CNN
- First 4 opening moves are random for diversity
- Fixed delay between AI turns for watchability
- Useful for comparing model behavior

### 5.6 Gameplay Flow (Anvil)

1. User logs in and opens **Game Setup**.
2. User selects model (`CNN` or `Transformer`) and side (`Player 1` or `Player 2`).
3. On the **Game Board**, user clicks a column to drop a chip.
4. Frontend sends board state + player + model choice to backend.
5. Backend returns `recommended_move`; frontend animates AI move and updates turn state.
6. Game ends on win/draw and exposes rematch or settings options.

### 5.7 Technical Implementation

**Dependencies:**
- TensorFlow/Keras for model inference
- NumPy for array operations
- Anvil Uplink SDK for server communication

**Model Loading:**
- Transformer: Weights loaded into rebuilt architecture
- CNN: Full model loaded directly from .h5 file
- Graceful handling when model files are missing

---

## Appendix A: Project Structure

```
connect-4/
├── anvil_webpage/
│   ├── LoginPage.html / .py
│   ├── GameSetup.html / .py
│   ├── GameBoard.html / .py
│   └── ModelReport.html / .py
├── aws_docker_backend/
│   ├── aws_backend.py
│   ├── model_wrappers.py
│   ├── connect4_engine.py
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
├── network_training/
│   ├── CNN/
│   │   ├── cnn_builder.ipynb
│   │   └── Using the CNN readme.txt
│   └── Transformer/
│       ├── transformer_colab.ipynb
│       └── transformer.weights.h5
├── data_generation/
│   └── data_generation_no_guide.ipynb  # MCTS data generation
├── data_balance/
│   └── balance.py                  # Move-depth loss weighting
├── mirroring/
│   └── mirror.py                   # Horizontal data augmentation
├── Connect4_Demo/
│   └── play_connect4.ipynb         # Interactive gameplay demo
├── data_viewer.ipynb               # Dataset exploration
└── README.md
```

## Appendix B: Data Pipeline Flow

```
MCTS Self-Play (2000 simulations/move)
           │
           v
    703K unique positions
           │
           v
  Horizontal Mirroring (mirror.py)
           │
           v
    1.4M augmented samples
           │
           v
  Move-Depth Weighting (balance.py)
           │
           v
    Sample weights array
           │
     ┌─────┴─────┐
     │           │
     v           v
   CNN      Transformer
 Training    Training
     │           │
     v           v
final_supervised_256f.h5   transformer.weights.h5
     │           │
     └─────┬─────┘
           │
           v
    AWS Backend / Anvil App
           │
           v
     Interactive Play
```

## Appendix C: Resources

| Resource | Link |
|----------|------|
| Data Generation Code | [Google Drive](https://drive.google.com/file/d/1tq9ASdZPYUzMduKU36TS4skNxAv2MqJb/view?usp=sharing) |
| Dataset (npz) | [GitHub Release v0.1-data](https://github.com/AHMerrill/connect-4/releases/tag/v0.1-data) |
| CNN Model & Code | [Google Drive](https://drive.google.com/drive/folders/1qzKsT7cwXddIeu15dXuG_vK7q4rD_PnG?usp=sharing) |
| GitHub Repository | [AHMerrill/connect-4](https://github.com/AHMerrill/connect-4) |
