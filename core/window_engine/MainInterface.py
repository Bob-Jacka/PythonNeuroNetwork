import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton

from core.window_engine.elements.Button import Button


class Interface:
    app: QApplication
    win: QMainWindow
    button: Button

    def __init__(self, x, y, length, width, windowTitle: str = 'Neuro window'):
        self.app = QApplication(sys.argv)
        self.win = QMainWindow()
        self.win.setGeometry(x, y, length, width)
        self.win.setWindowTitle(windowTitle)
        self.win.show()
        sys.exit(self.app.exec_())
