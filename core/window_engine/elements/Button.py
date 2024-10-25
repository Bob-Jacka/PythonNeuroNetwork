from PyQt5.QtCore import QRect
from PyQt5.QtWidgets import QPushButton, QWidget


class Button:
    button: QPushButton
    button_text: str

    def __init__(self, connectToView: QWidget, geometry: QRect = QRect(100, 100, 100, 100),
                 button_text: str = 'button', ):
        self.button = QPushButton(connectToView)
        self.button_text = button_text
        self.button.setGeometry(geometry)

    @classmethod
    def get_button_text(cls):
        return cls.button_text

    @classmethod
    def click_on(cls):
        cls.button.click()
