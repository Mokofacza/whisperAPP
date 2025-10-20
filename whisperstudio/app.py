# --- Hotfixy zgodności muszą zostać załadowane PRZED importami bibliotek audio/GUI ---
from .core.compat import apply_numpy_compat
from .core.soundcard_compat import patch_soundcard_numpy_fromstring

# 1) globalny shim: binarny fromstring -> frombuffer (bezpiecznie dla NumPy 2.x)
apply_numpy_compat()
# 2) targetowany hotfix dla soundcard.mediafoundation (Windows loopback)
patch_soundcard_numpy_fromstring()

import sys
from PySide6 import QtWidgets
from .ui.main_window import MainWindow

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
