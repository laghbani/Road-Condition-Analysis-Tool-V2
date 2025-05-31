from pathlib import Path
import tempfile
import json

import folium
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl

class MapWidget(QWebEngineView):
    """Zeigt OSM-Karte mit GPS-Track + Heading-Pfeilen."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tmp_html = Path(tempfile.mkstemp(suffix=".html")[1])
        self._map = folium.Map(tiles="OpenStreetMap")
        self._drawn = False
        self._refresh()

    def set_data(self, gps_df, rot_mat):
        if gps_df is None or gps_df.empty:
            self._map = folium.Map(tiles="OpenStreetMap")
            self._refresh()
            return

        lat0 = gps_df["lat"].iloc[0]
        lon0 = gps_df["lon"].iloc[0]
        self._map = folium.Map(location=[lat0, lon0], zoom_start=16, tiles="OpenStreetMap")

        folium.PolyLine(gps_df[["lat", "lon"]].values, color="blue", weight=2).add_to(self._map)

        if rot_mat is not None:
            import numpy as np
            R = np.array(rot_mat)
            fwd = R @ np.array([1, 0, 0])
            heading_deg = (np.degrees(np.arctan2(fwd[1], fwd[0])) + 360) % 360
            folium.Marker(
                location=[lat0, lon0],
                icon=folium.plugins.BeautifyIcon(
                    icon_shape="arrow",
                    border_color="#d35400",
                    border_width=2,
                    text_color="#d35400",
                    icon_rotate=heading_deg,
                ),
                tooltip=f"Heading ≈ {heading_deg:.1f}°",
            ).add_to(self._map)

        self._refresh()

    def _refresh(self):
        self._map.save(self._tmp_html)
        self.setUrl(QUrl.fromLocalFile(str(self._tmp_html)))
