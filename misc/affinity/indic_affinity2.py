import sys
import os
import re
import glob
import math
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtGui import QPainter, QColor, QFont, QPen
from PySide6.QtCore import Qt, QRect


# ==========================================
# 1. LOGIC LAYER: Data Scaling
# ==========================================
class AccuracyScaler:
    def __init__(self, logbase=2):
        self.logbase = logbase
        # Standardize range: log(0+1) to log(100+1)
        self.max_log = math.log(101, self.logbase)

    def get_intensity(self, value):
        """Maps accuracy [0-100] to [0.0-1.0] using log(acc + 1)."""
        val_clamped = max(0.0, min(value, 100.0))
        current_log = math.log(val_clamped + 1, self.logbase)
        return current_log / self.max_log


# ==========================================
# 2. VIEW LAYER: PySide6 Visualization
# ==========================================
class AffinityHeatmap(QWidget):
    def __init__(self, rdict, scaler, kordr=None):
        super().__init__()
        self.rdict = rdict
        self.scaler = scaler
        self.setWindowTitle("Affinity Matrix: Log-Scaled Accuracy")

        # Remove margins and constraints
        self.setContentsMargins(0, 0, 0, 0)

        # Logic for keys (using the refactored kordr logic)
        all_tr = set(rdict.keys())
        all_te = set()
        for te_map in rdict.values():
            all_te.update(te_map.keys())

        if kordr:
            self.tr_keys = [k for k in kordr if k in all_tr] + sorted(list(all_tr - set(kordr)))
            self.te_keys = [k for k in kordr if k in all_te] + sorted(list(all_te - set(kordr)))
        else:
            self.tr_keys = sorted(list(all_tr))
            self.te_keys = sorted(list(all_te))

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # 1. Force a bright background (White)
        p.fillRect(self.rect(), Qt.white)

        m_left, m_top = 120, 100
        grid_w = self.width() - m_left
        grid_h = self.height() - m_top

        if not self.tr_keys:
            p.setPen(Qt.black)  # Black text for errors
            p.drawText(self.rect(), Qt.AlignCenter, "No data parsed from logs.")
            return

        cw = grid_w / len(self.te_keys)
        ch = grid_h / len(self.tr_keys)

        for i, tr in enumerate(self.tr_keys):
            # 2. Draw Y-Axis Labels (Black)
            p.setPen(Qt.black)
            p.drawText(QRect(5, m_top + i * ch, m_left - 10, ch),
                       Qt.AlignRight | Qt.AlignCenter, tr)

            for j, te in enumerate(self.te_keys):
                # 3. Draw X-Axis Labels (Black)
                if i == 0:
                    p.save()
                    p.setPen(Qt.black)
                    p.translate(m_left + j * cw + cw / 2, m_top - 5)
                    p.rotate(-45)
                    p.drawText(0, 0, te)
                    p.restore()

                # Get value and scale intensity
                acc = self.rdict.get(tr, {}).get(te, 0.0)
                intensity = self.scaler.get_intensity(acc)

                # Color Mapping (HslF: hue, saturation, lightness)
                hue = 0.66 * (1.0 - intensity)
                color = QColor.fromHslF(hue, 0.8, 0.6)  # Slightly lighter color cells

                # Draw Rect
                rect = QRect(m_left + j * cw, m_top + i * ch, math.ceil(cw), math.ceil(ch))
                p.fillRect(rect, color)

                # 4. Cell Border (Light Grey/Translucent Black)
                p.setPen(QPen(QColor(0, 0, 0, 30), 0.5))
                p.drawRect(rect)

                # 5. Draw Accuracy Text (Contrast logic)
                if cw > 40 and ch > 20:
                    # If intensity is high (dark blue), use white text; otherwise black
                    p.setPen(Qt.white if intensity > 0.6 else Qt.black)
                    p.drawText(rect, Qt.AlignCenter, f"{acc:.1f}")
# ==========================================
# 3. PARSING LAYER: Original Logic
# ==========================================
def grep_a(file_path, pattern, after_lines=0):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        match_lines = []
        for i, line in enumerate(lines):
            if re.search(pattern, line):
                match_lines.extend(lines[i:i + 1 + after_lines])
        return match_lines
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []


def mkdict(line):
    terms = line.split(",")
    d = {}
    for t in terms:
        parts = t.split(":")  # Original used t[0], assuming key:val
        if len(parts) == 2:
            d[parts[0].strip()] = parts[1].strip()
    return d


def handle_f(rdict, file_path, epok=2, itrk=0):
    patt = "istrtest.*Epoch: " + str(epok) + ".*" + "Iter: " + str(itrk) + ".*ACR"
    lines = grep_a(file_path, patt)

    # Extract training script name from path
    # Path structure assumed: .../something-something-TRAINSCRIPT/log.txt
    try:
        tr_script = os.path.basename(os.path.dirname(file_path)).split("-")[2]
    except IndexError:
        tr_script = os.path.basename(os.path.dirname(file_path))

    if tr_script not in rdict:
        rdict[tr_script] = {}

    for l in lines:
        d = mkdict(l)
        if "TEST" in d and "ACR" in d:
            te_script = d["TEST"].split("_")[1]
            te_acr = float(d["ACR"]) * 100
            rdict[tr_script][te_script] = te_acr
    return rdict


# ==========================================
# 4. EXECUTION
# ==========================================
def main():
    ROOT = "/run/media/lasercat/writebuffer/320-eccv-IA/dump/watch_and_build/raw/Alvis/"
    rd = {}

    # Find all log files
    log_files = glob.glob(os.path.join(ROOT, "*", "*.log"))

    if not log_files:
        print(f"No log files found in {ROOT}")

    for f in log_files:
        rd = handle_f(rd, f)

    # Launch PySide6 UI
    app = QApplication(sys.argv)
    kordr=["hindi", "kannada", "malayalam", "marathi", "odia", "telugu","punjabi","tamil","bengali", "gujarati"]
    scaler = AccuracyScaler(logbase=2)
    viewer = AffinityHeatmap(rd, scaler,kordr)
    viewer.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()