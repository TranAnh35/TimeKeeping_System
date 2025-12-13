# Configuration

Runtime configuration for TimeKeeping_System.

Edit `src/config/config.json` and restart for changes to take effect.

## Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `COOLDOWN_SECONDS` | int | 300 | Seconds between check-in/out for same person |
| `HOLD_TIME_SECONDS` | float | 1.5 | Seconds face must be held to confirm |
| `RECOGNITION_THRESHOLD` | float | 0.55 | Distance threshold for face matching |
| `ENABLE_WEB_SERVER` | bool | true | Enable web dashboard at port 5000 |
| `WEB_PORT` | int | 5000 | Web server port |
| `FORCE_GUI_MODE` | bool | false | Force video window on Pi |
| `ENABLE_CENTER_ROI` | bool | true | Only detect faces in center region |
| `CENTER_ROI_RATIO` | float | 0.6 | Size of center ROI (60% of frame) |
| `ENABLE_MIDNIGHT_CHECKOUT` | bool | true | Auto checkout at 00:00 |
| `GC_INTERVAL` | int | 900 | Frames between garbage collection |
| `ENABLE_ADAPTIVE_SKIP` | bool/null | null | Auto-enable on Pi if null |
| `TARGET_PROCESS_TIME` | float | 0.15 | Target time per frame (adaptive skip) |
| `MIN_FRAME_SKIP` | int | 1 | Minimum frames to skip |
| `MAX_FRAME_SKIP` | int | 5 | Maximum frames to skip |
| `DEFAULT_FRAME_SKIP` | int/null | null | Fixed skip count (null = auto) |
| `CAMERA_WIDTH` | int/null | null | Camera width (null = auto) |
| `CAMERA_HEIGHT` | int/null | null | Camera height (null = auto) |
| `TFLITE_NUM_THREADS` | int/null | null | TFLite threads (null = auto) |

## Notes

- Values set to `null` will use platform-specific defaults
- Pi automatically uses lower resolution and more threads
- Web dashboard: `http://<IP>:5000`
- Management UI: `http://<IP>:5000/manage`
