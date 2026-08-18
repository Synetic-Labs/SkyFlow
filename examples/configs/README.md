# Example drone config (CLI text)

One example set, for reference. Real per-drone configs live with the
trainer that owns the drone.

- `air75_dump.txt` — full CLI `dump all` from a physical drone (BETAFPV
  Air75, Betaflight 4.5.0). The durable config format.
- `sim_overrides.txt` — CLI lines appended after the dump: sim-only
  transport changes and rename repairs. Last value wins.
- `known_rejects.txt` — committed reject snapshot for the cudaflight
  firmware build. The render fails when the actual reject list differs
  from this file in either direction.

Render the eeprom image at use time and pass it to a fleet
(needs cudaflight >= 0.3.5):

```python
from pathlib import Path
import tempfile

import cudaflight
from skyflow.firmware import GpuFirmwareFleet

base = Path(__file__).parent
image = cudaflight.render_eeprom(
    base / "air75_dump.txt",
    base / "sim_overrides.txt",
    known_rejects=base / "known_rejects.txt",
)
with tempfile.NamedTemporaryFile(suffix=".bin") as f:
    f.write(image)
    f.flush()
    fw = GpuFirmwareFleet(4096, eeprom=f.name)
```

Never commit a rendered `.bin`. An image one parameter-group version
behind the firmware does not fail at boot — Betaflight factory-resets
the whole config, silently, and the fleet flies stock defaults. The
render path fails loudly instead: any CLI line the firmware rejects
outside the committed snapshot aborts the render and names the line.
