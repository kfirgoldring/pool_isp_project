"""
Billiards Assistance System — Entry Point
==========================================
Launches the GUI. All pipeline control (hardware selection, calibration,
detection, trajectory) is managed inside the GUI itself.
"""

import sys


def main():
    try:
        from PyQt5.QtWidgets import QApplication
    except ImportError:
        print('ERROR: PyQt5 is not installed.  Run:  pip install PyQt5')
        sys.exit(1)

    from gui import BilliardsApp

    app = QApplication(sys.argv)
    app.setApplicationName('Billiards Assistance System')
    window = BilliardsApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
