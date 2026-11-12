"""End-to-end checks of the Streamlit app with Streamlit's own test harness.

These start the real solver worker process, so they take a little while:
    python -m pytest tests/test_app.py -q
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parent.parent / "app.py")
PRESETS = [
    "Evening peak (default)",
    "Severe shortfall",
    "Residential already off 2 h today",
    "Hospital protected (9 qubits)",
]


def plan(preset: str, layers: int = 1, noisy: bool = False, fairness: float | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.sidebar.selectbox[0].select(preset)
    at.run()
    layer_slider = [s for s in at.sidebar.slider if s.label.startswith("QAOA layers")][0]
    layer_slider.set_value(layers)
    if noisy:
        at.sidebar.toggle[0].set_value(True)
    if fairness is not None:
        at.sidebar.select_slider[0].set_value(fairness)
    at.run()
    at.sidebar.button[0].click()
    at.run()
    return at


def page_text(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


@pytest.mark.parametrize("preset", PRESETS)
def test_every_preset_plans_a_valid_schedule(preset):
    at = plan(preset)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    text = page_text(at)
    assert "Schedule to publish" in text
    assert "Every hour" in text  # shortfall covered in every hour


def test_default_scenario_matches_the_exact_optimum():
    at = plan("Evening peak (default)", layers=2)
    assert not at.exception, [e.value for e in at.exception]
    assert "Optimal ✓" in page_text(at)


def test_fairness_off_reproduces_the_priority_list():
    at = plan("Evening peak (default)", fairness=0.0)
    assert not at.exception, [e.value for e in at.exception]
    assert any("Fairness is off" in i.value for i in at.info)


def test_noise_option_still_returns_a_schedule():
    at = plan("Evening peak (default)", noisy=True)
    assert not at.exception, [e.value for e in at.exception]
    assert "Schedule to publish" in page_text(at)
