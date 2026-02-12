# ============================================
# GAME BOARD PAGE - LONGHORNS VERSION
# With Rematch Button and AI vs AI Mode
# Paste this in the Code tab for GameBoard
# ============================================
from ._anvil_designer import GameBoardTemplate
from anvil import *
import anvil.server
import random
from anvil.js.window import setTimeout

class GameBoard(GameBoardTemplate):
    def __init__(self, username="Player", model_type="CNN", player_number=1, ai_vs_ai=False, **properties):
        self.init_components(**properties)

        # Game settings
        self.username = username
        self.model_type = model_type
        self.player_number = player_number
        self.ai_vs_ai = ai_vs_ai  # AI vs AI mode flag

        # AI vs AI settings
        self.random_opening_moves = 4  # First 4 moves are random
        self.move_count = 0
        self.ai_move_delay = 800  # ms between AI moves

        if ai_vs_ai:
            # In AI vs AI mode: CNN is orange (player 1), Transformer is white (player 2)
            # Randomly decide who goes first
            if random.choice([True, False]):
                self.orange_model = "CNN"
                self.white_model = "Transformer"
            else:
                self.orange_model = "Transformer"
                self.white_model = "CNN"
            self.current_turn = "orange"  # Orange always starts
        else:
            # Normal player vs AI mode
            if player_number == 1:
                self.human_color = "orange"
                self.ai_color = "white"
                self.human_value = 1
                self.ai_value = -1
            else:
                self.human_color = "white"
                self.ai_color = "orange"
                self.human_value = -1
                self.ai_value = 1

        # Game state
        self.board = [[0 for _ in range(7)] for _ in range(6)]
        self.game_over = False
        self.is_human_turn = (player_number == 1) if not ai_vs_ai else False

    def form_show(self, **event_args):
        """Called when form is shown"""
        from anvil.js.window import document
        self.doc = document

        # Set game info
        game_info = self.doc.getElementById('game-info')
        if game_info:
            if self.ai_vs_ai:
                game_info.textContent = f"CNN vs Transformer"
            else:
                game_info.textContent = f"{self.username} vs {self.model_type}"

        # Setup AI vs AI banner
        if self.ai_vs_ai:
            banner = self.doc.getElementById('ai-vs-ai-banner')
            if banner:
                banner.classList.add('visible')
            # Hide hover row in AI vs AI mode
            hover_row = self.doc.getElementById('hover-row')
            if hover_row:
                hover_row.classList.add('hidden')
        else:
            # Set hover chip colors for human player
            for i in range(7):
                hover_chip = self.doc.getElementById(f'hover-{i}')
                if hover_chip:
                    hover_chip.classList.add(self.human_color)

            # Setup click handlers for columns (only in human mode)
            hover_cells = self.doc.querySelectorAll('.hover-cell')
            for cell in hover_cells:
                col = int(cell.getAttribute('data-col'))
                def make_click_handler(column):
                    def handler(event):
                        self.column_click(column)
                    return handler
                cell.addEventListener('click', make_click_handler(col))

            board_cells = self.doc.querySelectorAll('.cell')
            for cell in board_cells:
                col = int(cell.getAttribute('data-col'))
                def make_click_handler(column):
                    def handler(event):
                        self.column_click(column)
                    return handler
                cell.addEventListener('click', make_click_handler(col))

        # Header buttons
        new_game_btn = self.doc.getElementById('btn-new-game')
        quit_btn = self.doc.getElementById('btn-quit')

        if new_game_btn:
            new_game_btn.addEventListener('click', lambda e: open_form('GameSetup', username=self.username))
        if quit_btn:
            quit_btn.addEventListener('click', lambda e: open_form('LoginPage'))

        # Game over buttons
        rematch_btn = self.doc.getElementById('btn-rematch')
        change_settings_btn = self.doc.getElementById('btn-change-settings')

        if rematch_btn:
            rematch_btn.addEventListener('click', lambda e: self.rematch())
        if change_settings_btn:
            change_settings_btn.addEventListener('click', lambda e: open_form('GameSetup', username=self.username))

        # Update turn display
        self.update_turn_display()

        # Start the game
        if self.ai_vs_ai:
            # Start AI vs AI battle
            setTimeout(self.make_ai_vs_ai_move, 1000)
        elif not self.is_human_turn:
            # AI goes first in human vs AI
            setTimeout(self.make_ai_move, 1000)

    def rematch(self):
        """Start a new game with the same settings"""
        if self.ai_vs_ai:
            open_form('GameBoard',
                      username=self.username,
                      model_type=self.model_type,
                      player_number=self.player_number,
                      ai_vs_ai=True)
        else:
            open_form('GameBoard',
                      username=self.username,
                      model_type=self.model_type,
                      player_number=self.player_number,
                      ai_vs_ai=False)

    def column_click(self, col):
        """Handle column click (human player only)"""
        if self.game_over or not self.is_human_turn or self.ai_vs_ai:
            return

        row = self.get_lowest_empty_row(col)
        if row == -1:
            return

        self.make_move(row, col, self.human_value, self.human_color)
        self.move_count += 1

        if self.check_win(row, col, self.human_value):
            self.end_game("human")
            return

        if self.is_board_full():
            self.end_game("draw")
            return

        self.is_human_turn = False
        self.update_turn_display()
        setTimeout(self.make_ai_move, 800)

    def get_lowest_empty_row(self, col):
        """Find lowest empty row"""
        for row in range(5, -1, -1):
            if self.board[row][col] == 0:
                return row
        return -1

    def make_move(self, row, col, value, color):
        """Place a chip"""
        self.board[row][col] = value

        chip = self.doc.getElementById(f'chip-{row}-{col}')
        if chip:
            chip.classList.add(color)
            chip.classList.add('dropping')

            def remove_drop():
                chip.classList.remove('dropping')
            setTimeout(remove_drop, 500)

    def make_ai_move(self):
        """AI makes a move (human vs AI mode)"""
        if self.game_over:
            return

        status = self.doc.getElementById('status-message')
        if status:
            status.textContent = f"{self.model_type} is thinking..."

        try:
            current_player = 'plus' if self.ai_value == 1 else 'minus'
            response = anvil.server.call('process_move', {
                'board': self.board,
                'current_player': current_player,
                'model_type': self.model_type
            })
            ai_col = response.get('recommended_move', self.get_fallback_move())
        except:
            ai_col = self.get_fallback_move()

        row = self.get_lowest_empty_row(ai_col)
        if row != -1:
            self.make_move(row, ai_col, self.ai_value, self.ai_color)
            self.move_count += 1

            if self.check_win(row, ai_col, self.ai_value):
                self.end_game("ai")
                return

            if self.is_board_full():
                self.end_game("draw")
                return

        self.is_human_turn = True
        self.update_turn_display()

        if status:
            status.textContent = ""

    def make_ai_vs_ai_move(self):
        """AI vs AI: Make a move for the current AI"""
        if self.game_over:
            return

        current_model = self.orange_model if self.current_turn == "orange" else self.white_model
        current_color = self.current_turn
        current_value = 1 if current_color == "orange" else -1

        status = self.doc.getElementById('status-message')

        # Determine if this is a random opening move
        if self.move_count < self.random_opening_moves:
            if status:
                status.textContent = f"Random opening move {self.move_count + 1}/{self.random_opening_moves}..."
            valid_cols = [c for c in range(7) if self.board[0][c] == 0]
            ai_col = random.choice(valid_cols) if valid_cols else 3
        else:
            if status:
                status.textContent = f"{current_model} is thinking..."
            try:
                current_player = 'plus' if current_value == 1 else 'minus'
                response = anvil.server.call('process_move', {
                    'board': self.board,
                    'current_player': current_player,
                    'model_type': current_model
                })
                ai_col = response.get('recommended_move', self.get_fallback_move())
            except:
                ai_col = self.get_fallback_move()

        row = self.get_lowest_empty_row(ai_col)
        if row != -1:
            self.make_move(row, ai_col, current_value, current_color)
            self.move_count += 1

            if self.check_win(row, ai_col, current_value):
                self.end_game(current_model)
                return

            if self.is_board_full():
                self.end_game("draw")
                return

        # Switch turns
        self.current_turn = "white" if self.current_turn == "orange" else "orange"
        self.update_turn_display()

        if status:
            status.textContent = ""

        # Schedule next AI move
        setTimeout(self.make_ai_vs_ai_move, self.ai_move_delay)

    def get_fallback_move(self):
        """Random valid move with center preference"""
        valid = [c for c in range(7) if self.board[0][c] == 0]
        for c in [3, 2, 4, 1, 5, 0, 6]:
            if c in valid:
                return c
        return random.choice(valid) if valid else 3

    def check_win(self, row, col, player):
        """Check for 4 in a row"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        winning_cells = []

        for dr, dc in directions:
            cells = [(row, col)]
            count = 1

            r, c = row + dr, col + dc
            while 0 <= r < 6 and 0 <= c < 7 and self.board[r][c] == player:
                cells.append((r, c))
                count += 1
                r, c = r + dr, c + dc

            r, c = row - dr, col - dc
            while 0 <= r < 6 and 0 <= c < 7 and self.board[r][c] == player:
                cells.append((r, c))
                count += 1
                r, c = r - dr, c - dc

            if count >= 4:
                winning_cells = cells
                break

        if winning_cells:
            for (r, c) in winning_cells:
                chip = self.doc.getElementById(f'chip-{r}-{c}')
                if chip:
                    chip.classList.add('winner')
            return True

        return False

    def is_board_full(self):
        """Check if board is full"""
        return all(self.board[0][c] != 0 for c in range(7))

    def update_turn_display(self):
        """Update turn indicator"""
        if self.game_over:
            return

        indicator = self.doc.getElementById('turn-indicator')
        if not indicator:
            return

        if self.ai_vs_ai:
            current_model = self.orange_model if self.current_turn == "orange" else self.white_model
            indicator.textContent = f"🤖 {current_model}'S TURN"
            indicator.classList.remove('your-turn', 'ai-turn', 'cnn-turn', 'transformer-turn')
            if self.current_turn == "orange":
                indicator.classList.add('cnn-turn')
            else:
                indicator.classList.add('transformer-turn')
        else:
            if self.is_human_turn:
                indicator.textContent = "🤘 YOUR TURN"
                indicator.classList.remove('ai-turn')
                indicator.classList.add('your-turn')
            else:
                indicator.textContent = f"🤖 {self.model_type}'S TURN"
                indicator.classList.remove('your-turn')
                indicator.classList.add('ai-turn')

    def end_game(self, winner):
        """Handle game end"""
        self.game_over = True

        indicator = self.doc.getElementById('turn-indicator')
        status = self.doc.getElementById('status-message')
        game_over_buttons = self.doc.getElementById('game-over-buttons')

        if status:
            status.textContent = ""

        if game_over_buttons:
            game_over_buttons.classList.add('visible')

        if indicator:
            indicator.classList.remove('your-turn', 'ai-turn', 'cnn-turn', 'transformer-turn')

            if self.ai_vs_ai:
                if winner == "draw":
                    indicator.textContent = "🤝 IT'S A DRAW!"
                    indicator.classList.add('draw')
                else:
                    indicator.textContent = f"🏆 {winner.upper()} WINS!"
                    indicator.classList.add('winner')
            else:
                if winner == "human":
                    indicator.textContent = "🤘 HOOK 'EM! YOU WIN!"
                    indicator.classList.add('winner')
                elif winner == "ai":
                    indicator.textContent = f"🤖 {self.model_type} WINS"
                    indicator.classList.add('loser')
                else:
                    indicator.textContent = "🤝 IT'S A DRAW"
                    indicator.classList.add('draw')
