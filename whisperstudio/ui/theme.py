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
* {{ font-family: "Segoe UI", "Inter", "Noto Sans", sans-serif; font-size: 10pt; }}
QMainWindow {{ background: {BG.name()}; }}
QStatusBar {{ color: {SUBTLE.name()}; border-top: 1px solid {BORDER.name()}; padding: 4px 8px; }}

/* --- Sidebar --- */
QFrame#Sidebar {{
  background: #0b0d12; border-right: 1px solid {BORDER.name()};
}}
QPushButton.SideItem {{
  background: transparent; color: {TEXT.name()};
  border: 0px; text-align: left; padding: 12px 14px; border-radius: 10px;
  icon-size: 18px; /* Rozmiar ikony w sidebarze */
}}
QPushButton.SideItem:hover {{ background: #121620; }}
QPushButton.SideItem[active="true"] {{
    background: #172035; color: {ACCENT.name()}; font-weight: 600;
}}

/* --- Karty (użyj .setProperty("class", "Card")) --- */
QFrame[class~="Card"] {{
  background: #0a0d14; border: 1px solid {BORDER.name()}; border-radius: 14px;
}}

/* --- Kontenery --- */
QGroupBox {{
  color: {TEXT.name()};
  border: 1px solid {BORDER.name()};
  border-radius: 12px; margin-top: 12px; padding: 10px;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; }}

/* --- Kontrolki --- */
QLabel {{ color: {TEXT.name()}; padding: 2px; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {{
  background: {CARD.name()}; color: {TEXT.name()};
  border: 1px solid {BORDER.name()}; border-radius: 10px; padding: 8px 10px;
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {{
  border: 1px solid {ACCENT.name()};
}}
QComboBox::drop-down {{ border: 0px; }}

/* --- Przyciski --- */
QPushButton {{
  background: #1b2330; color: {TEXT.name()};
  border: 1px solid {BORDER.name()}; border-radius: 10px; padding: 10px 16px;
  font-weight: 600;
  icon-size: 16px;
}}
QPushButton:hover {{ background: #1f2937; }}
QPushButton:disabled {{ color: #7a7f8a; background: #141822; }}

QPushButton[class~="Primary"] {{ background: {SUCCESS.name()}; color: white; border: 0; }}
QPushButton[class~="Primary"]:hover {{ background: #37b34e; }}
QPushButton[class~="Primary"]:disabled {{ background: #335a46; color: #94b8a2; }}
QPushButton[class~="Danger"]  {{ background: {DANGER.name()}; color: white; border: 0; }}
QPushButton[class~="Danger"]:hover {{ background: #e04f4f; }}
QPushButton[class~="Danger"]:disabled {{ background: #693a3a; color: #c09a9a; }}

/* ToolButton (np. 'Up' w pickerze) */
QToolButton {{
  background: #1b2330; color: {TEXT.name()};
  border: 1px solid {BORDER.name()}; border-radius: 10px; padding: 8px;
  icon-size: 16px;
}}
QToolButton:hover {{ background: #1f2937; }}

/* --- Inne --- */
QProgressBar {{
  background: #141822; color: {TEXT.name()};
  border: 1px solid {BORDER.name()}; border-radius: 10px; height: 10px; /* Cieńszy */
  text-align: center;
}}
QProgressBar::chunk {{ background: {SUCCESS.name()}; border-radius: 10px; }}

QTabWidget::pane {{
  border: 1px solid {BORDER.name()}; border-radius: 14px; padding: 6px; background: #0c0f15;
}}
QTabBar::tab {{
  background: transparent; color: {SUBTLE.name()};
  padding: 10px 16px; border-radius: 10px; margin: 4px; font-weight: 600;
}}
QTabBar::tab:selected {{ color: {TEXT.name()}; background: #182030; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #2b3546; border-radius: 5px; min-height: 28px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #2b3546; border-radius: 5px; min-width: 28px; }}
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