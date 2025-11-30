Configuration for TimeKeeping_System
==================================

You can configure runtime behavior without changing source code using the JSON file at `config/config.json`.

Key values:
- `COOLDOWN_SECONDS` (int): Seconds between two attendance registrations from the same person (default: 300)
- `HOLD_TIME_SECONDS` (float): Seconds a face must be held before registration (default: 1.5)
- `ENABLE_WEB_SERVER` (bool): Show the web dashboard (default: true)
- `WEB_PORT` (int): Port for the dashboard (default: 5000)
- `ENABLE_ANTISPOOF` (bool): Toggle anti-spoofing (default: false)
- `RECOGNITION_THRESHOLD` (float): Distance threshold for recognition (default: 0.55)
- `FORCE_GUI_MODE` (bool): Force showing video window on Pi (default: false)
- `ENABLE_MIDNIGHT_CHECKOUT` (bool): Enable nightly auto-checkout at 00:00 (default: true)
- `LOW_MEMORY_MODE` (bool or null): If `null`, auto-detected on Pi; else override (default: null)
- `CAMERA_WIDTH/HEIGHT` (int or null): Camera resolution; if `null`, auto computed
- `ENABLE_ADAPTIVE_SKIP` (bool or null): If `null`, will be enabled on Pi
- `TARGET_PROCESS_TIME` (float), `MIN_FRAME_SKIP`, `MAX_FRAME_SKIP`, `DEFAULT_FRAME_SKIP`

Edit `config/config.json` and restart the program for changes to take effect.
