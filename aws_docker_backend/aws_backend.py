"""
Connect 4 AWS Backend - UPDATED VERSION
Uses real CNN and Transformer models
"""
import anvil.server
import numpy as np
import os
import random

# Import model wrappers for real model predictions
from model_wrappers import get_cnn_move, get_transformer_move, load_cnn_model, load_transformer_model

# Track if models are loaded
models_loaded = {
    'cnn': False,
    'transformer': False
}

def initialize_models():
    """Initialize and test both models on startup"""
    global models_loaded
    print("=" * 50)
    print("🚀 Connect 4 Backend - Initializing Models")
    print("=" * 50)

    # Test CNN model
    try:
        cnn = load_cnn_model()
        if cnn is not None:
            models_loaded['cnn'] = True
            print("✅ CNN model loaded successfully")
        else:
            print("⚠️ CNN model not available - will use fallback")
    except Exception as e:
        print(f"❌ CNN model failed to load: {e}")

    # Test Transformer model
    try:
        transformer = load_transformer_model()
        if transformer is not None:
            models_loaded['transformer'] = True
            print("✅ Transformer model loaded successfully")
        else:
            print("⚠️ Transformer model not available - will use fallback")
    except Exception as e:
        print(f"❌ Transformer model failed to load: {e}")

    print("=" * 50)
    print(f"📊 Model Status: CNN={'Real' if models_loaded['cnn'] else 'Fallback'}, "
          f"Transformer={'Real' if models_loaded['transformer'] else 'Fallback'}")
    print("=" * 50)


def get_fallback_move(board):
    """Fallback move when models are unavailable - prefers center columns"""
    valid_columns = [col for col in range(7) if board[0, col] == 0]

    if not valid_columns:
        return 3

    # Prefer center columns
    for col in [3, 2, 4, 1, 5, 0, 6]:
        if col in valid_columns:
            return col

    return random.choice(valid_columns)


@anvil.server.callable
def check_connection():
    """
    Simple connection test function for Anvil Uplink.
    Returns a confirmation string when the server is online.
    """
    cnn_status = "Real" if models_loaded['cnn'] else "Fallback"
    transformer_status = "Real" if models_loaded['transformer'] else "Fallback"
    return f'AWS Server Online! Models: CNN={cnn_status}, Transformer={transformer_status}'


@anvil.server.callable
def process_move(board_data):
    """
    Process a Connect 4 move using the selected AI model.

    Args:
        board_data: Dictionary containing:
            - board: 6x7 board state (1=plus, -1=minus, 0=empty)
            - current_player: 'plus', 'minus', 1, or 2
            - model_type: 'CNN' or 'Transformer'

    Returns:
        Dictionary with AI response and move recommendation
    """
    try:
        # Convert board to numpy array
        board = np.array(board_data['board'])
        current_player = board_data['current_player']
        model_type = board_data.get('model_type', 'CNN')

        print(f"🎮 Processing move - Player: {current_player}, Model: {model_type}")
        print(f"📋 Board state:\n{board}")

        # Normalize current_player to 'plus' or 'minus'
        # Handle both string ('plus'/'minus') and numeric (1/2) formats
        if current_player in ['minus', 2, -1]:
            is_minus_player = True
            current_player_str = 'minus'
        else:
            is_minus_player = False
            current_player_str = 'plus'

        # Apply symmetry rule if AI is playing as minus
        # Models are trained to play as 'plus' (1), so we flip the board
        if is_minus_player:
            print("🔄 Applying symmetry flip (board * -1) for minus player")
            board = board * -1
            print(f"📋 Flipped board state:\n{board}")

        # Get AI move based on model type
        using_real_model = False

        if model_type == 'CNN':
            if models_loaded['cnn']:
                print("🤖 Using Real CNN Model")
                move = get_cnn_move(board)
                using_real_model = True
            else:
                print("🤖 Using Fallback (CNN not loaded)")
                move = get_fallback_move(board)

        elif model_type == 'Transformer':
            if models_loaded['transformer']:
                print("🤖 Using Real Transformer Model")
                move = get_transformer_move(board)
                using_real_model = True
            else:
                print("🤖 Using Fallback (Transformer not loaded)")
                move = get_fallback_move(board)
        else:
            print(f"⚠️ Unknown model type '{model_type}', using CNN")
            if models_loaded['cnn']:
                move = get_cnn_move(board)
                using_real_model = True
            else:
                move = get_fallback_move(board)

        # Validate move is within bounds
        if move is None or move < 0 or move > 6:
            print(f"❌ Invalid move {move}, using fallback")
            move = get_fallback_move(board)

        # Validate move is legal (column not full)
        # Note: Check against original board orientation for minus player
        original_board = np.array(board_data['board'])
        if original_board[0, move] != 0:
            print(f"⚠️ Move {move} is illegal (column full), choosing valid move")
            move = get_fallback_move(original_board)

        print(f"✅ Final recommended move: {move}")

        return {
            'status': 'success',
            'recommended_move': int(move),
            'model_used': model_type,
            'board_processed': True,
            'symmetry_applied': is_minus_player,
            'using_real_model': using_real_model
        }

    except Exception as e:
        print(f"❌ Error processing move: {str(e)}")
        import traceback
        traceback.print_exc()

        # Fallback response
        fallback_move = 3
        try:
            original_board = np.array(board_data['board'])
            fallback_move = get_fallback_move(original_board)
        except:
            pass

        return {
            'status': 'error',
            'error': str(e),
            'recommended_move': fallback_move,
            'model_used': board_data.get('model_type', 'CNN'),
            'board_processed': False,
            'symmetry_applied': False,
            'using_real_model': False
        }


if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Connect 4 AWS Backend Starting...")
    print("=" * 50)

    # Initialize models first
    initialize_models()

    # Get uplink key from environment variable
    uplink_key = os.environ.get("ANVIL_UPLINK_KEY", "").strip()

    if not uplink_key:
        print("❌ ANVIL_UPLINK_KEY is missing!")
        print("   Set it in .env file or environment variable")
        exit(1)

    print(f"🔑 Using uplink key: {uplink_key[:20]}...")

    # Connect to Anvil
    try:
        anvil.server.connect(uplink_key)
        print("✅ Connected to Anvil successfully!")
        print("🎮 Backend ready - waiting for game requests...")
        print("=" * 50)

        # Keep the server running
        anvil.server.wait_forever()

    except Exception as e:
        print(f"❌ Failed to connect to Anvil: {e}")
        exit(1)
