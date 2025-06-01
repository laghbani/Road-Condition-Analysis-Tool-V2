# Road-Condition-Analysis-Tool-V2
This tool visualises IMU data and now supports ISO 2631-1/VDI 2057 frequency weighting.
Weighted signals and their running RMS can be toggled in the GUI.
Comfort or health weighting modes colour the RMS according to the following classes:

| Klasse | awv [m/s²] | Farbe |
|-------|-----------|-------|
| Comfortable | < 1.72 | grün |
| Slightly uncomfortable | 1.72–2.12 | gelb |
| Uncomfortable | 2.12–2.54 | orange |
| Very uncomfortable | 2.54–3.19 | rot |
| Extremely uncomfortable | ≥ 3.19 | violett |
