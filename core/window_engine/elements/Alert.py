from PyQt5.QtCore import QRect
from PyQt5.QtWidgets import QMessageBox, QWidget


class Alert:
    alert: QMessageBox
    title: str
    text: str

    def __init__(self, connectToView: QWidget, geometry: QRect = QRect(100, 100, 100, 100)):
        self.alert = QMessageBox(connectToView)
        self.alert.setGeometry(geometry)
