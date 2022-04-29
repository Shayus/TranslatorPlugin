import sys
from translate import Translator
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5 import QtCore
import UI
import paddle
import paddle.nn.functional as F
import re
import numpy as np


class MainDialog(QDialog):
    def __init__(self, parent=None):
        super(QDialog, self).__init__(parent)
        self.ui = UI.Ui_Dialog()
        self.ui.setupUi(self)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

    def to_chinese(self):
        word = self.ui.textEdit.toPlainText()
        translator = Translator(to_lang="chinese")
        translation = translator.translate(word)
        self.ui.textBrowser_2.setText(translation)



if __name__ == '__main__':
    print(paddle.__version__)
    app = QApplication(sys.argv)
    dlg = MainDialog()
    dlg.show()
    sys.exit(app.exec_())
