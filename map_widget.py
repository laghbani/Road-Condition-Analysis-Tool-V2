from pathlib import Path
import tempfile
import json

import folium
from folium.plugins import BeautifyIcon
from geopy.distance import geodesic
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, Qt
from PyQt5.QtWidgets import QMenu

class MapWidget(QWebEngineView):
    """Zeigt OSM-Karte mit GPS-Track + Heading-Pfeilen."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tmp_html = Path(tempfile.mkstemp(suffix=".html")[1])
        self._map = folium.Map(tiles="OpenStreetMap")
        self._drawn = False
        self._gps_df = None
        self._rot_mat = None
        self._refresh()

    def set_data(self, gps_df, rot_mat, every_n=200):
        self._gps_df = gps_df
        self._rot_mat = rot_mat
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
            idxs = range(0, len(gps_df), every_n) if len(gps_df) > every_n else [0]
            for j in idxs:
                lat, lon = gps_df.loc[j, ["lat", "lon"]]
                fwd = R @ np.array([1, 0, 0])
                heading = (np.degrees(np.arctan2(fwd[1], fwd[0])) + 360) % 360
                self._draw_arrow(lat, lon, heading, L=6)

        self._refresh()

    def _draw_arrow(self, lat, lon, heading, L=6):
        pt = geodesic(meters=L).destination((lat, lon), heading)
        folium.PolyLine(
            [[lat, lon], [pt.latitude, pt.longitude]],
            color="orange", weight=2
        ).add_to(self._map)

    def _refresh(self):
        self._map.save(self._tmp_html)
        self.setUrl(QUrl.fromLocalFile(str(self._tmp_html)))

    def mousePressEvent(self, ev):
        if ev.button() == Qt.RightButton:
            m = QMenu(self)
            m.addAction(
                "Hide sensor→vehicle matrix",
                lambda: self.set_data(self._gps_df, None),
            )
            m.exec(ev.globalPos())
        else:
            super().mousePressEvent(ev)
