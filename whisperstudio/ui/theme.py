# -*- coding: utf-8 -*-
from __future__ import annotations
from PySide6 import QtCore, QtGui, QtWidgets

ACCENT = QtGui.QColor("#6ea8fe")      # niebieski akcent
BG      = QtGui.QColor("#0f1115")     # tło
CARD    = QtGui.QColor("#141822")     # karty/pola
BORDER  = QtGui.QColor("#2a2f3a")     # obramowanie
TEXT    = QtGui.QColor("#e8eaf0")     # tekst
SUBTLE  = QtGui.QColor("#c6c9d3")     # opis
DANGER  = QtGui.QColor("#d03b3b")     # czerwony
SUCCESS = QtGui.QColor("#2ea043")     # zielony

def build_palette() -> QtGui.QPalette:
    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.Window, BG)
    pal.setColor(QtGui.QPalette.Base, CARD)
    pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#0a0d14"))
    pal.setColor(QtGui.QPalette.Text, TEXT)
    pal.setColor(QtGui.QPalette.ButtonText, TEXT)
    pal.setColor(QtGui.QPalette.WindowText, TEXT)
    pal.setColor(QtGui.QPalette.ToolTipBase, CARD)
    pal.setColor(QtGui.QPalette.ToolTipText, TEXT)
    pal.setColor(QtGui.QPalette.Highlight, ACCENT)
    pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#0b0d12"))
    pal.setColor(QtGui.QPalette.Button, CARD)
    return pal

_QSS = f"""
* {{ font-family: "Segoe UI", "Inter", "Noto Sans", sans-serif; }}
QMainWindow {{ background: {BG.name()}; }}
QStatusBar {{ color: {SUBTLE.name()}; border-top: 1px solid {BORDER.name()}; }}

QFrame#Sidebar {{
  background: #0b0d12; border-right: 1px solid {BORDER.name()};
}}
QPushButton.SideItem {{
  background: transparent; color: {TEXT.name()};
  border: 0px; text-align: left; padding: 10px 14px; border-radius: 10px;
}}
QPushButton.SideItem:hover {{ background: #121620; }}
QPushButton.SideItem[active="true"] {{ background: #172035; color: {ACCENT.name()}; }}

QGroupBox {{
  color: {TEXT.name()};
  border: 1px solid {BORDER.name()};
  border-radius: 12px; margin-top: 12px;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; }}

QLabel {{ color: {TEXT.name()}; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {{
  background: {CARD.name()}; color: {TEXT.name()};
  border: 1px solid {BORDER.name()}; border-radius: 10px; padding: 6px 8px;
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {{
  border: 1px solid {ACCENT.name()};
  box-shadow: 0 0 0 3px rgba(110,168,254,0.15);
}}

QPushButton {{
  background: #1b2330; color: {TEXT.name()};
  border: 1px solid {BORDER.name()}; border-radius: 10px; padding: 8px 14px;
}}
QPushButton:hover {{ background: #1f2937; }}
QPushButton:disabled {{ color: #7a7f8a; background: #141822; }}

QPushButton.Primary {{ background: {SUCCESS.name()}; color: white; border: 0; }}
QPushButton.Primary:disabled {{ background: #335a46; }}
QPushButton.Danger  {{ background: {DANGER.name()}; color: white; border: 0; }}

QProgressBar {{
  background: #141822; color: {TEXT.name()};
  border: 1px solid {BORDER.name()}; border-radius: 10px; height: 18px;
}}
QProgressBar::chunk {{ background: {SUCCESS.name()}; border-radius: 10px; }}

QTabWidget::pane {{
  border: 1px solid {BORDER.name()}; border-radius: 14px; padding: 6px; background: #0c0f15;
}}
QTabBar::tab {{
  background: transparent; color: {SUBTLE.name()};
  padding: 8px 14px; border-radius: 10px; margin: 4px;
}}
QTabBar::tab:selected {{ color: {TEXT.name()}; background: #182030; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #2b3546; border-radius: 5px; min-height: 28px; }}
"""

def apply_theme(app: QtWidgets.QApplication):
    # HiDPI
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app.setStyle("Fusion")
    app.setPalette(build_palette())
    app.setStyleSheet(_QSS)
