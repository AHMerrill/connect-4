"""
Connect 4 Engine Module
Board representation: 1 = Plus (Player 1), -1 = Minus (Player 2), 0 = Empty
"""

import numpy as np


def update_board(board_temp, color, column):
    """
    Updates the board by placing a piece in the specified column.

    Args:
        board_temp: Current 6x7 numpy array board state
        color: 'plus' or 'minus' player
        column: Column index (0-6) to place piece

    Returns:
        Updated board numpy array
    """
    board = board_temp.copy()

    # Find lowest empty row in column
    for row in range(5, -1, -1):
        if board[row, column] == 0:
            if color == 'plus':
                board[row, column] = 1
            else:
                board[row, column] = -1
            break

    return board


def check_for_win(board, col):
    """
    Checks for a win after a piece was placed in the specified column.

    Args:
        board: Current 6x7 numpy array board state
        col: Column index where last piece was placed

    Returns:
        String indicating winner: 'v-plus', 'v-minus', 'h-plus', 'h-minus',
        'd-plus', 'd-minus', or 'nobody'
    """
    # Find the row where the last piece was placed
    row = -1
    for r in range(6):
        if board[r, col] != 0:
            row = r
            break

    if row == -1:
        return 'nobody'

    player = board[row, col]

    # Check all directions
    directions = [
        (0, 1),   # Horizontal
        (1, 0),   # Vertical
        (1, 1),   # Diagonal down-right
        (1, -1),  # Diagonal down-left
    ]

    for dr, dc in directions:
        count = 1

        # Check positive direction
        r, c = row + dr, col + dc
        while 0 <= r < 6 and 0 <= c < 7 and board[r, c] == player:
            count += 1
            r, c = r + dr, c + dc

        # Check negative direction
        r, c = row - dr, col - dc
        while 0 <= r < 6 and 0 <= c < 7 and board[r, c] == player:
            count += 1
            r, c = r - dr, c - dc

        if count >= 4:
            if player == 1:
                return 'plus'
            else:
                return 'minus'

    return 'nobody'


def prepare_nn_input(board, player):
    """
    Prepares neural network input by applying symmetry transformation.

    Ensures the Neural Network only ever has to learn how to play as '1' (plus player).

    Args:
        board: Current 6x7 numpy array board state
        player: 'plus' (1) or 'minus' (-1) player

    Returns:
        Transformed board where current player is always represented as 1
    """
    if player == 'plus' or player == 1:
        return board.copy()
    elif player == 'minus' or player == -1 or player == 2:
        return board * -1
    else:
        raise ValueError("Player must be 'plus', 'minus', 1, -1, or 2")


def find_legal(board):
    """
    Finds all legal moves (columns that are not full).

    Args:
        board: Current 6x7 numpy array board state

    Returns:
        List of legal column indices
    """
    legal = [i for i in range(7) if board[0, i] == 0]
    return legal


def is_board_full(board):
    """Check if the board is completely full"""
    return all(board[0, col] != 0 for col in range(7))


def print_board(board):
    """Pretty print the board"""
    symbols = {0: '.', 1: 'X', -1: 'O'}
    print("\n  0 1 2 3 4 5 6")
    print("  -------------")
    for row in range(6):
        print(f"| ", end="")
        for col in range(7):
            print(f"{symbols[int(board[row, col])]} ", end="")
        print("|")
    print("  -------------")
