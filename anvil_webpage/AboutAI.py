# ============================================
# ABOUT AI PAGE - Neural Network Description
# Requirement #2: Detailed NN explanation + Board Analysis
# Paste this in the Code tab for AboutAI
# ============================================
from ._anvil_designer import AboutAITemplate
from anvil import *

class AboutAI(AboutAITemplate):
    def __init__(self, username="Player", **properties):
        self.init_components(**properties)
        self.username = username

        # Bind the show event
        self.add_event_handler('show', self.form_show)

    def form_show(self, **event_args):
        """Called when form is shown"""
        from anvil.js.window import document
        self.doc = document

        # Back button
        back_btn = self.doc.getElementById('btn-back')
        if back_btn:
            back_btn.addEventListener('click', lambda e: open_form('GameSetup', username=self.username))

        # Play button
        play_btn = self.doc.getElementById('btn-play')
        if play_btn:
            play_btn.addEventListener('click', lambda e: open_form('GameSetup', username=self.username))

        # Render example boards
        self.render_example_boards()

    def render_example_boards(self):
        """Render the example Connect 4 boards"""

        # Easy Example 1: Obvious Win (3 in a row, need to complete)
        # Orange about to win horizontally
        board_easy_1 = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0,-1,-1, 0, 0, 0],
            [0, 1, 1, 1, 0,-1, 0],  # Orange has 3 in a row, col 4 wins
        ]
        highlight_easy_1 = [(5, 4)]  # Winning move

        # Easy Example 2: Block Opponent
        # White has 3 in a row, orange must block
        board_easy_2 = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0],
            [0,-1,-1,-1, 0, 1, 0],  # White has 3, must block at col 4
        ]
        highlight_easy_2 = [(5, 4)]  # Blocking move

        # Hard Example 1: Complex mid-game
        board_hard_1 = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1,-1, 0, 0, 0],
            [0, 0,-1, 1, 1, 0, 0],
            [0, 1, 1,-1,-1, 0, 0],
            [1,-1, 1,-1, 1,-1, 0],
        ]
        highlight_hard_1 = [(4, 1)]  # Best move is tricky

        # Hard Example 2: Dual threats setup
        board_hard_2 = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1,-1, 0, 0, 0],
            [0,-1, 1,-1, 1, 0, 0],
            [-1, 1,-1, 1,-1, 1, 0],
        ]
        highlight_hard_2 = [(3, 4)]  # Fork setup

        # Render all boards
        self.render_board('board-easy-1', board_easy_1, highlight_easy_1, 'correct')
        self.render_board('board-easy-2', board_easy_2, highlight_easy_2, 'correct')
        self.render_board('board-hard-1', board_hard_1, highlight_hard_1, 'wrong')
        self.render_board('board-hard-2', board_hard_2, highlight_hard_2, 'wrong')

    def render_board(self, board_id, board, highlights, highlight_class):
        """Render a mini Connect 4 board"""
        container = self.doc.getElementById(board_id)
        if not container:
            return

        # Clear existing content
        container.innerHTML = ''

        # Create rows
        for row in range(6):
            row_div = self.doc.createElement('div')
            row_div.className = 'mini-row'

            for col in range(7):
                cell_div = self.doc.createElement('div')
                cell_div.className = 'mini-cell'

                # Set chip color
                if board[row][col] == 1:
                    cell_div.classList.add('orange')
                elif board[row][col] == -1:
                    cell_div.classList.add('white')

                # Add highlight for key moves
                if (row, col) in highlights:
                    cell_div.classList.add(highlight_class)

                row_div.appendChild(cell_div)

            container.appendChild(row_div)
