
import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

class VideoThread(QtCore.QThread):
    frameRaw = QtCore.pyqtSignal(np.ndarray)
    frameBOS = QtCore.pyqtSignal(np.ndarray)
    error    = QtCore.pyqtSignal(str)

    def __init__(self, rtsp_url):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.running = False

        # Processing parameters (defaults)
        self.filter_name        = "Gaussian Blur"
        self.param_value        = 5
        self.bg_update_interval = 0
        self.gain               = 1.0
        self.colormap           = None

        # Internal state
        self._background_gray = None
        self._frame_count     = 0

    def run(self):
        self.running = True
        cap = None
        while self.running:
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(self.rtsp_url)
                if not cap.isOpened():
                    self.error.emit("Unable to connect to stream. Retrying...")
                    QtCore.QThread.msleep(1000)
                    continue
                self.error.emit("")
                self._background_gray = None
                self._frame_count     = 0

            ret, frame = cap.read()
            if not ret:
                cap.release()
                cap = None
                continue

            frame_color = frame  # raw BGR frame
            if self._background_gray is None:
                self._background_gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
                continue

            gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
            if self.bg_update_interval > 0:
                self._frame_count += 1
                if self._frame_count >= self.bg_update_interval:
                    self._background_gray = gray.copy()
                    self._frame_count     = 0

            diff = cv2.absdiff(gray, self._background_gray)
            bos  = diff

            # Apply selected filter
            p = self.param_value
            try:
                if self.filter_name == "Gaussian Blur":
                    k = max(1, int(p)) | 1
                    bos = cv2.GaussianBlur(diff, (k, k), 0)
                elif self.filter_name == "Median Filter":
                    k = max(1, int(p)) | 1
                    bos = cv2.medianBlur(diff, k)
                elif self.filter_name == "Bilateral Filter":
                    sigma = max(1, int(p))
                    bos = cv2.bilateralFilter(diff, 9, sigma, sigma)
                elif self.filter_name == "Sobel Edges":
                    k = max(1, int(p)) | 1
                    k = min(k, 7)
                    sx = cv2.Sobel(diff, cv2.CV_64F, 1, 0, ksize=k)
                    sy = cv2.Sobel(diff, cv2.CV_64F, 0, 1, ksize=k)
                    mag = np.hypot(sx, sy)
                    bos = cv2.convertScaleAbs(mag)
                elif self.filter_name == "Laplacian Edges":
                    k = max(1, int(p)) | 1
                    k = min(k, 7)
                    lap = cv2.Laplacian(diff, cv2.CV_64F, ksize=k)
                    bos = cv2.convertScaleAbs(lap)
                elif self.filter_name == "Unsharp Masking":
                    amount = float(p) / 50.0
                    blur   = cv2.GaussianBlur(diff, (5, 5), 0)
                    high   = diff.astype(np.float32) - blur.astype(np.float32)
                    sharp  = diff.astype(np.float32) + amount * high
                    bos    = np.clip(sharp, 0, 255).astype(np.uint8)
                elif self.filter_name == "Ratio":
                    eps        = 1e-6
                    curr_f     = gray.astype(np.float32)
                    bg_f       = self._background_gray.astype(np.float32)
                    ratio_diff = np.abs(curr_f / (bg_f + eps) - 1.0)
                    bos        = np.clip(ratio_diff * 255 * float(p), 0, 255).astype(np.uint8)
            except:
                bos = diff

            # Apply gain
            bos_f = bos.astype(np.float32) * self.gain
            bos8  = np.clip(bos_f, 0, 255).astype(np.uint8)

            # Apply colormap
            if self.colormap is not None:
                bos_color = cv2.applyColorMap(bos8, self.colormap)
            else:
                bos_color = cv2.cvtColor(bos8, cv2.COLOR_GRAY2BGR)

            # Emit frames
            self.frameRaw.emit(frame_color)
            self.frameBOS.emit(bos_color)

        if cap:
            cap.release()

    def stop(self):
        self.running = False
        self.wait(1000)

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time BOS Viewer")

        # Layouts
        vbox = QtWidgets.QVBoxLayout(self)
        hbox = QtWidgets.QHBoxLayout()
        ctrl = QtWidgets.QGridLayout()
        vbox.addLayout(hbox)
        vbox.addLayout(ctrl)

        # Video display labels (expandable)
        self.raw_lbl = QtWidgets.QLabel()
        self.bos_lbl = QtWidgets.QLabel()
        for lbl in (self.raw_lbl, self.bos_lbl):
            lbl.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            lbl.setScaledContents(True)
            lbl.setAlignment(QtCore.Qt.AlignCenter)
        hbox.addWidget(self.raw_lbl)
        hbox.addWidget(self.bos_lbl)
        hbox.setStretch(0, 1)
        hbox.setStretch(1, 1)

        # Filter selector
        ctrl.addWidget(QtWidgets.QLabel("Filter:"), 0, 0)
        self.filter_combo = QtWidgets.QComboBox()
        self.filter_combo.addItems([
            "Gaussian Blur", "Median Filter", "Bilateral Filter",
            "Sobel Edges", "Laplacian Edges", "Unsharp Masking", "Ratio"
        ])
        ctrl.addWidget(self.filter_combo, 0, 1)

        # Parameter slider
        ctrl.addWidget(QtWidgets.QLabel("Param:"), 1, 0)
        self.param_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.param_slider.setRange(1, 21)
        self.param_slider.setValue(5)
        self.param_value_label = QtWidgets.QLabel("5")
        ctrl.addWidget(self.param_slider, 1, 1)
        ctrl.addWidget(self.param_value_label, 1, 2)

        # Background update interval
        ctrl.addWidget(QtWidgets.QLabel("BG Update (frames):"), 2, 0)
        self.bg_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.bg_slider.setRange(0, 500)
        self.bg_slider.setValue(0)
        self.bg_value_label = QtWidgets.QLabel("0 (off)")
        ctrl.addWidget(self.bg_slider, 2, 1)
        ctrl.addWidget(self.bg_value_label, 2, 2)

        # Gain slider (0.5x–10x)
        ctrl.addWidget(QtWidgets.QLabel("Gain:"), 3, 0)
        self.gain_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gain_slider.setRange(5, 100)
        self.gain_slider.setValue(10)
        self.gain_slider.setTickInterval(5)
        self.gain_value_label = QtWidgets.QLabel("1.0")
        ctrl.addWidget(self.gain_slider, 3, 1)
        ctrl.addWidget(self.gain_value_label, 3, 2)

        # Colormap selector
        ctrl.addWidget(QtWidgets.QLabel("Colormap:"), 4, 0)
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(["None", "JET", "HOT", "BONE", "VIRIDIS", "INFERNO"])
        ctrl.addWidget(self.cmap_combo, 4, 1)

        # Status label
        self.status_label = QtWidgets.QLabel("")
        ctrl.addWidget(self.status_label, 5, 0, 1, 3)

        # Start video thread
        self.thread = VideoThread("rtsp://10.5.0.2:8554/mystream")
        self.thread.frameRaw.connect(self.update_raw)
        self.thread.frameBOS.connect(self.update_bos)
        self.thread.error.connect(self.on_error)
        self.thread.start()

        # Connect controls
        self.filter_combo.currentTextChanged.connect(self.on_filter)
        self.param_slider.valueChanged.connect(self.on_param)
        self.bg_slider.valueChanged.connect(self.on_bg)
        self.gain_slider.valueChanged.connect(self.on_gain)
        self.cmap_combo.currentTextChanged.connect(self.on_cmap)

        # Initialize control states
        self.on_filter(self.filter_combo.currentText())
        self.on_param(self.param_slider.value())
        self.on_bg(self.bg_slider.value())
        self.on_gain(self.gain_slider.value())
        self.on_cmap(self.cmap_combo.currentText())

    @QtCore.pyqtSlot(np.ndarray)
    def update_raw(self, frame):
        h, w, ch = frame.shape
        img = QtGui.QImage(frame.data, w, h, ch * w, QtGui.QImage.Format_BGR888)
        self.raw_lbl.setPixmap(QtGui.QPixmap.fromImage(img))

    @QtCore.pyqtSlot(np.ndarray)
    def update_bos(self, frame):
        h, w, ch = frame.shape
        img = QtGui.QImage(frame.data, w, h, ch * w, QtGui.QImage.Format_BGR888)
        self.bos_lbl.setPixmap(QtGui.QPixmap.fromImage(img))

    @QtCore.pyqtSlot(str)
    def on_error(self, msg):
        self.status_label.setText(msg)

    def on_filter(self, text):
        self.thread.filter_name = text
        # adjust slider range if needed
        if text in ["Gaussian Blur", "Median Filter"]:
            self.param_slider.setRange(1, 21)
            self.param_slider.setTickInterval(2)
        elif text == "Bilateral Filter":
            self.param_slider.setRange(1, 100)
            self.param_slider.setTickInterval(10)
        elif text in ["Sobel Edges", "Laplacian Edges"]:
            self.param_slider.setRange(1, 7)
            self.param_slider.setTickInterval(2)
        elif text == "Unsharp Masking":
            self.param_slider.setRange(0, 100)
            self.param_slider.setTickInterval(10)
        elif text == "Ratio":
            self.param_slider.setRange(1, 100)
            self.param_slider.setTickInterval(10)
        self.on_param(self.param_slider.value())

    def on_param(self, v):
        self.param_value_label.setText(str(v))
        self.thread.param_value = v

    def on_bg(self, v):
        self.bg_value_label.setText(f"{v} (off)" if v == 0 else str(v))
        self.thread.bg_update_interval = v

    def on_gain(self, v):
        gain = v / 10.0
        self.gain_value_label.setText(f"{gain:.1f}")
        self.thread.gain = gain

    def on_cmap(self, name):
        cmap_map = {
            "None":    None,
            "JET":     cv2.COLORMAP_JET,
            "HOT":     cv2.COLORMAP_HOT,
            "BONE":    cv2.COLORMAP_BONE,
            "VIRIDIS": cv2.COLORMAP_VIRIDIS,
            "INFERNO": cv2.COLORMAP_INFERNO,
        }
        self.thread.colormap = cmap_map.get(name)

    def closeEvent(self, event):
        self.thread.stop()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    win.resize(1200, 700)
    sys.exit(app.exec_())
