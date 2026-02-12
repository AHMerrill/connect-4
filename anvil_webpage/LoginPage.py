# ============================================
# LOGIN PAGE - SIMPLE VERSION
# This should work with Anvil Custom HTML
# ============================================
from ._anvil_designer import LoginPageTemplate
from anvil import *
import anvil.server

class LoginPage(LoginPageTemplate):
    def __init__(self, **properties):
        self.init_components(**properties)
        # Valid credentials
        self.valid_users = {
            "demo": "demo",
            "player1": "connect4",
            "admin": "admin123",
            "dan": "Optimization1234"
        }

    def form_show(self, **event_args):
        """Called when the form is shown on screen"""
        # Import here to ensure DOM is ready
        from anvil.js.window import document

        # Get elements
        self.username_input = document.getElementById('username')
        self.password_input = document.getElementById('password')
        self.error_msg = document.getElementById('error-msg')
        self.login_btn = document.getElementById('login-btn')

        # Attach click handler
        if self.login_btn:
            self.login_btn.addEventListener('click', self.do_login)

        # Attach enter key handler
        if self.password_input:
            self.password_input.addEventListener('keyup', self.check_enter)

    def check_enter(self, event):
        """Check if Enter was pressed"""
        if event.key == 'Enter':
            self.do_login(event)

    def do_login(self, event):
        """Perform login"""
        username = self.username_input.value.strip().lower() if self.username_input else ""
        password = self.password_input.value if self.password_input else ""

        # Validate
        if not username or not password:
            self.error_msg.textContent = "Please enter both username and password"
            return

        # Check credentials
        if username in self.valid_users and self.valid_users[username] == password:
            self.error_msg.textContent = ""
            open_form('GameSetup', username=username)
        else:
            self.error_msg.textContent = "Invalid username or password"
            self.password_input.value = ""
