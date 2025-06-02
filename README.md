# Road-Condition-Analysis-Tool-V2

This repository includes `iso_weighting.py` implementing ISO 2631‑1 / VDI 2057
frequency weighting utilities for vibration analysis. The GUI (`main_gui_v2.py`)
can display weighted acceleration and detected peaks. Enable *Show ISO weighted*
from the **View** menu. Comfort and health weighting can now be selected via
**IMU Settings → Comfort/Health mode weighting**. The map tab shows an
interactive OpenStreetMap powered by *folium*. Current settings can be stored
and later restored using **File → Save settings…** and **File → Load settings…**.
Window geometry and active topics are automatically preserved between sessions
via Qt's `QSettings`.
