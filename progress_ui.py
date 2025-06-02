from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QListWidget,
    QProgressBar, QPushButton, QApplication,
)
from PyQt5.QtCore import Qt

class ProgressWindow(QDialog):
    """Mehrstufiger Fortschrittsdialog mit Prozent-Anzeige und Detail-Liste."""

    def __init__(self, title: str, steps: list[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.ApplicationModal)
        self.resize(380, 240)

        vbox = QVBoxLayout(self)

        self.lbl_step = QLabel("Initialisiere …")
        vbox.addWidget(self.lbl_step)

        self.bar = QProgressBar()
        self.total_steps = len(steps)
        self.bar.setRange(0, self.total_steps)
        self.bar.setValue(0)
        vbox.addWidget(self.bar)

        self.lst = QListWidget()
        self.lst.addItems(steps)
        for i in range(self.lst.count()):
            it = self.lst.item(i)
            it.setFlags(Qt.ItemIsEnabled)
        self.lst.setCurrentRow(0)
        vbox.addWidget(self.lst, stretch=1)

        self.btn_abort = QPushButton("Abbrechen")
        self.btn_abort.clicked.connect(self.reject)
        vbox.addWidget(self.btn_abort)

        self.show()

    def advance(self, text: str | None = None) -> bool:
        """Fortschritt +1 Step.  Rückgabe: True = nicht abgebrochen."""
        if text:
            self.lbl_step.setText(text)
        new_val = self.bar.value() + 1
        self.bar.setValue(new_val)
        self.lst.setCurrentRow(new_val)
        QApplication.processEvents()
        return not self.wasCanceled()

    # ------------------------------------------------------------------
    def set_bar_range(self, maximum: int) -> None:
        """Extern den Fortschrittsbalken neu skalieren."""
        self.bar.setMaximum(maximum)
        QApplication.processEvents()

    def set_bar_value(self, value: int) -> None:
        """Extern den Fortschritt setzen."""
        self.bar.setValue(value)
        QApplication.processEvents()

    def set_bar_steps(self, step: int) -> None:
        """Zurück auf Schritt-basierten Fortschritt setzen."""
        self.bar.setMaximum(self.total_steps)
        self.bar.setValue(step)
        QApplication.processEvents()

    def wasCanceled(self) -> bool:
        return self.result() == QDialog.Rejected
