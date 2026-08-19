# Example drone config (CLI text)

- `stock_dump.txt` — the stock `dump all` of the pinned cudaflight
  firmware build (Betaflight 2026.6.1 SITL defaults). A placeholder
  example: keep real per-drone configs with the trainer that owns the
  drone, not in this repo.

Config source of truth is CLI text. The eeprom `.bin` is a derived
artifact: render it at use time and never commit it. Rendering is strict
— any line the firmware rejects fails the render and is named.

Render and boot a fleet (needs cudaflight >= 0.5.0):

```python
from pathlib import Path
import tempfile

import cudaflight
from skyflow.firmware import GpuFirmwareFleet

image = cudaflight.render_eeprom(Path(__file__).parent / "stock_dump.txt")
with tempfile.NamedTemporaryFile(suffix=".bin") as f:
    f.write(image)
    f.flush()
    fw = GpuFirmwareFleet(4096, eeprom=f.name)
```

Why never commit a `.bin`: an image one parameter-group version behind
the firmware does not fail at boot — Betaflight factory-resets the whole
config, silently, and the fleet flies stock defaults.
