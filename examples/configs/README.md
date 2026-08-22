# Example drone config (CLI text)

- `stock_dump.txt` — the stock `dump all` of the pinned cudaflight
  firmware build (Betaflight 2026.6.1 SITL defaults). A placeholder
  example: keep real per-drone configs with the trainer that owns the
  drone, not in this repo.

Config source of truth is CLI text. The eeprom `.bin` is a derived
artifact: render it at use time and never commit it. Rendering is strict
— any line the firmware rejects fails the render and is named.

The env renders for you — point `SimConfig.eeprom` at the dump
(`control="sticks"`, needs cudaflight >= 0.5.0):

```python
from skyflow import SimConfig, SkyFlowEnv

env = SkyFlowEnv(SimConfig(
    num_envs=4096,
    control="sticks",
    eeprom="examples/configs/stock_dump.txt",
    eeprom_overrides="my_sim_overrides.txt",  # optional sim-only CLI lines
))
print(env.eeprom_image)  # the rendered boot image (temp file), for run logs
```

Both fleet backends receive the same image; `examples/fly_drone_config.py`
is the end-to-end demo. `eeprom_overrides` holds sim-only pins a real
drone's dump needs on the SITL build (for example `set blackbox_device =
NONE` — the sim has no SPI flash chip). The stock dump needs none.

Why never commit a `.bin`: an image one parameter-group version behind
the firmware does not fail at boot — Betaflight factory-resets the whole
config, silently, and the fleet flies stock defaults. The render's
version gate turns that silence into a construction-time error.
