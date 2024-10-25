from PyQt5.QtCore import QRect
from PyQt5.QtWidgets import QLineEdit, QWidget


class TextView:
    textView: QLineEdit

    def __init__(self, connectToView: QWidget, geometry: QRect = QRect(100, 100, 100, 100)):
        self.textView = QLineEdit(connectToView)
        self.textView.setGeometry(geometry)
