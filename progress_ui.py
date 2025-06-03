from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QListWidget,
    QProgressBar, QPushButton, QApplication,
)
from PyQt5.QtCore import Qt

class ProgressWindow(QDialog):
    """Multi step progress dialog with percentage display and detail list."""

    def __init__(self, title: str, steps: list[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.ApplicationModal)
        self.resize(380, 240)

        vbox = QVBoxLayout(self)

        self.lbl_step = QLabel("Initializing …")
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

        self._aborted = False
        self.btn_abort = QPushButton("Abort")
        self.btn_abort.clicked.connect(self._on_abort)
        vbox.addWidget(self.btn_abort)

        self.show()

    def _on_abort(self) -> None:
        self._aborted = True
        self.reject()

    def advance(self, text: str | None = None) -> bool:
        """Advance by one step.  Returns ``True`` if not aborted."""
        if text:
            self.lbl_step.setText(text)
        new_val = self.bar.value() + 1
        self.bar.setValue(new_val)
        self.lst.setCurrentRow(new_val)
        QApplication.processEvents()
        return not self.wasCanceled()

    # ------------------------------------------------------------------
    def set_bar_range(self, maximum: int) -> None:
        """Externally adjust the progress bar range."""
        self.bar.setMaximum(maximum)
        QApplication.processEvents()

    def set_bar_value(self, value: int) -> None:
        """Set progress bar value from outside."""
        self.bar.setValue(value)
        QApplication.processEvents()

    def set_bar_steps(self, step: int) -> None:
        """Switch back to step-based progress."""
        self.bar.setMaximum(self.total_steps)
        self.bar.setValue(step)
        QApplication.processEvents()

    def wasCanceled(self) -> bool:
        return self._aborted
