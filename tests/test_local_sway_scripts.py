from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_sway_adapter_uses_constant_lock_polling() -> None:
    script = (REPO / "scripts" / "enable_local_doorman_sway_adapter.sh").read_text()

    assert "DOORMAN_SWAY_MAX_INTERVAL" not in script
    assert "delay=" not in script
    assert 'sleep "$interval"' in script


def test_sway_graphics_wrapper_pins_glvnd_and_dri_to_system_mesa() -> None:
    script = (REPO / "scripts" / "repair_local_sway_graphics.sh").read_text()

    assert "__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json" in script
    assert "LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri" in script
    assert "GBM_BACKENDS_PATH=/usr/lib/x86_64-linux-gnu/gbm" in script
