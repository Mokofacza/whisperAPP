from PySide6 import QtWidgets
from .ui.main_window import MainWindow

def main():
    app = QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    app.exec()
