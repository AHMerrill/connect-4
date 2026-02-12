# ============================================
# GAME SETUP PAGE - Python Code
# With Modal Support for How to Play, About Models, Credits
# Paste this in the Code tab for GameSetup
# ============================================
from ._anvil_designer import GameSetupTemplate
from anvil import *
import anvil.server

class GameSetup(GameSetupTemplate):
    def __init__(self, username="Player", **properties):
        self.init_components(**properties)
        self.username = username
        self.selected_model = None
        self.selected_player = None

    def form_show(self, **event_args):
        """Called when the form is shown"""
        from anvil.js.window import document
        self.doc = document

        # Set welcome text
        welcome = self.doc.getElementById('welcome-text')
        if welcome:
            welcome.textContent = f"Welcome, {self.username}!"

        # Get elements
        self.cnn_card = self.doc.getElementById('card-cnn')
        self.trans_card = self.doc.getElementById('card-transformer')
        self.p1_card = self.doc.getElementById('card-player1')
        self.p2_card = self.doc.getElementById('card-player2')
        self.start_btn = self.doc.getElementById('start-btn')
        self.status_text = self.doc.getElementById('status-text')

        # Add click handlers for model/player selection
        if self.cnn_card:
            self.cnn_card.addEventListener('click', self.select_cnn)
        if self.trans_card:
            self.trans_card.addEventListener('click', self.select_transformer)
        if self.p1_card:
            self.p1_card.addEventListener('click', self.select_player1)
        if self.p2_card:
            self.p2_card.addEventListener('click', self.select_player2)
        if self.start_btn:
            self.start_btn.addEventListener('click', self.start_game)

        # AI vs AI button
        ai_vs_ai_btn = self.doc.getElementById('btn-ai-vs-ai')
        if ai_vs_ai_btn:
            ai_vs_ai_btn.addEventListener('click', self.start_ai_vs_ai)

        # Setup modal handlers
        self.setup_modals()

        # Check server status
        self.check_server()

    def setup_modals(self):
        """Setup modal open/close handlers"""
        # Footer buttons to open modals
        btn_how_to_play = self.doc.getElementById('btn-how-to-play')
        btn_about_models = self.doc.getElementById('btn-about-models')
        btn_credits = self.doc.getElementById('btn-credits')

        if btn_how_to_play:
            btn_how_to_play.addEventListener('click', lambda e: self.open_modal('modal-how-to-play'))
        if btn_about_models:
            btn_about_models.addEventListener('click', lambda e: self.open_modal('modal-about-models'))
        if btn_credits:
            btn_credits.addEventListener('click', lambda e: self.open_modal('modal-credits'))

        # Close buttons
        close_how_to_play = self.doc.getElementById('close-how-to-play')
        close_about_models = self.doc.getElementById('close-about-models')
        close_credits = self.doc.getElementById('close-credits')

        if close_how_to_play:
            close_how_to_play.addEventListener('click', lambda e: self.close_modal('modal-how-to-play'))
        if close_about_models:
            close_about_models.addEventListener('click', lambda e: self.close_modal('modal-about-models'))
        if close_credits:
            close_credits.addEventListener('click', lambda e: self.close_modal('modal-credits'))

        # Close on overlay click
        modals = ['modal-how-to-play', 'modal-about-models', 'modal-credits']
        for modal_id in modals:
            modal = self.doc.getElementById(modal_id)
            if modal:
                def make_overlay_handler(m_id):
                    def handler(e):
                        if e.target.id == m_id:
                            self.close_modal(m_id)
                    return handler
                modal.addEventListener('click', make_overlay_handler(modal_id))

    def open_modal(self, modal_id):
        """Open a modal by ID"""
        modal = self.doc.getElementById(modal_id)
        if modal:
            modal.classList.add('active')

    def close_modal(self, modal_id):
        """Close a modal by ID"""
        modal = self.doc.getElementById(modal_id)
        if modal:
            modal.classList.remove('active')

    def check_server(self):
        """Check if AWS backend is online"""
        try:
            response = anvil.server.call('check_connection')
            if self.status_text:
                self.status_text.textContent = f"✅ {response}"
                self.status_text.classList.add('online')
        except:
            if self.status_text:
                self.status_text.textContent = "⚠️ Server offline - Demo mode"
                self.status_text.classList.add('offline')

    def select_cnn(self, event):
        self.selected_model = "CNN"
        self.update_ui()

    def select_transformer(self, event):
        self.selected_model = "Transformer"
        self.update_ui()

    def select_player1(self, event):
        self.selected_player = 1
        self.update_ui()

    def select_player2(self, event):
        self.selected_player = 2
        self.update_ui()

    def update_ui(self):
        """Update card selections"""
        # Model cards
        if self.cnn_card:
            if self.selected_model == "CNN":
                self.cnn_card.classList.add('selected')
            else:
                self.cnn_card.classList.remove('selected')

        if self.trans_card:
            if self.selected_model == "Transformer":
                self.trans_card.classList.add('selected')
            else:
                self.trans_card.classList.remove('selected')

        # Player cards
        if self.p1_card:
            if self.selected_player == 1:
                self.p1_card.classList.add('selected')
            else:
                self.p1_card.classList.remove('selected')

        if self.p2_card:
            if self.selected_player == 2:
                self.p2_card.classList.add('selected')
            else:
                self.p2_card.classList.remove('selected')

        # Start button
        if self.start_btn:
            if self.selected_model and self.selected_player:
                self.start_btn.disabled = False
            else:
                self.start_btn.disabled = True

    def start_game(self, event):
        """Start the game"""
        if self.selected_model and self.selected_player:
            open_form('GameBoard',
                      username=self.username,
                      model_type=self.selected_model,
                      player_number=self.selected_player,
                      ai_vs_ai=False)

    def start_ai_vs_ai(self, event):
        """Start AI vs AI battle"""
        open_form('GameBoard',
                  username=self.username,
                  model_type="CNN",
                  player_number=1,
                  ai_vs_ai=True)
