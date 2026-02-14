# ============================================
# MODEL REPORT PAGE - Neural Network Training Report
# With Tabbed Navigation
# Paste this in the Code tab for ModelReport
# ============================================
from ._anvil_designer import ModelReportTemplate
from anvil import *

class ModelReport(ModelReportTemplate):
    def __init__(self, username="Player", **properties):
        self.init_components(**properties)
        self.username = username

        # Bind the show event
        self.add_event_handler('show', self.form_show)

    def form_show(self, **event_args):
        """Called when form is shown"""
        from anvil.js.window import document
        self.doc = document

        # Setup tab navigation
        self.setup_tabs()

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

    def setup_tabs(self):
        """Setup tab navigation click handlers"""
        tab_btns = self.doc.querySelectorAll('.tab-btn')

        for btn in tab_btns:
            def make_tab_handler(tab_name):
                def handler(e):
                    self.switch_tab(tab_name)
                return handler
            tab_name = btn.getAttribute('data-tab')
            btn.addEventListener('click', make_tab_handler(tab_name))

    def switch_tab(self, tab_name):
        """Switch to the specified tab"""
        # Remove active class from all tabs and content
        tab_btns = self.doc.querySelectorAll('.tab-btn')
        tab_contents = self.doc.querySelectorAll('.tab-content')

        for btn in tab_btns:
            btn.classList.remove('active')
        for content in tab_contents:
            content.classList.remove('active')

        # Add active class to selected tab and content
        selected_btn = self.doc.querySelector(f'.tab-btn[data-tab="{tab_name}"]')
        selected_content = self.doc.getElementById(f'tab-{tab_name}')

        if selected_btn:
            selected_btn.classList.add('active')
        if selected_content:
            selected_content.classList.add('active')

    def render_example_boards(self):
        """Render the example Connect 4 boards"""

        # Easy Example 1: Obvious Win (3 in a row, need to complete)
        board_easy_1 = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0,-1,-1, 0, 0, 0],
            [0, 1, 1, 1, 0,-1, 0],
        ]
        highlight_easy_1 = [(5, 4)]

        # Easy Example 2: Block Opponent
        board_easy_2 = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0],
            [0,-1,-1,-1, 0, 1, 0],
        ]
        highlight_easy_2 = [(5, 4)]

        # Hard Example 1: Complex mid-game
        board_hard_1 = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1,-1, 0, 0, 0],
            [0, 0,-1, 1, 1, 0, 0],
            [0, 1, 1,-1,-1, 0, 0],
            [1,-1, 1,-1, 1,-1, 0],
        ]
        highlight_hard_1 = [(4, 1)]

        # Hard Example 2: Fork setup
        board_hard_2 = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1,-1, 0, 0, 0],
            [0,-1, 1,-1, 1, 0, 0],
            [-1, 1,-1, 1,-1, 1, 0],
        ]
        highlight_hard_2 = [(3, 4)]

        # Render all boards
        self.render_board('board-easy-1', board_easy_1, highlight_easy_1, 'highlight')
        self.render_board('board-easy-2', board_easy_2, highlight_easy_2, 'highlight')
        self.render_board('board-hard-1', board_hard_1, highlight_hard_1, 'wrong')
        self.render_board('board-hard-2', board_hard_2, highlight_hard_2, 'wrong')

    def render_board(self, board_id, board, highlights, highlight_class):
        """Render a mini Connect 4 board"""
        container = self.doc.getElementById(board_id)
        if not container:
            return

        container.innerHTML = ''

        for row in range(6):
            row_div = self.doc.createElement('div')
            row_div.className = 'mini-row'

            for col in range(7):
                cell_div = self.doc.createElement('div')
                cell_div.className = 'mini-cell'

                if board[row][col] == 1:
                    cell_div.classList.add('orange')
                elif board[row][col] == -1:
                    cell_div.classList.add('white')

                if (row, col) in highlights:
                    cell_div.classList.add(highlight_class)

                row_div.appendChild(cell_div)

            container.appendChild(row_div)
