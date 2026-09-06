import threading

from openpilot.common import realtime
from opendbc.sunnypilot.car.honda import longitudinal


class RecordingParams:
  instances = []
  events = []

  def __init__(self):
    self.values = {}
    self.instances.append(self)

  def put(self, key, value, block=False):
    self.values[key] = value
    self.events.append(("put", key, block))


def test_persistence_runs_off_thread(monkeypatch):
  started = threading.Event()
  release = threading.Event()
  calls = []
  RecordingParams.events.clear()

  monkeypatch.setattr(realtime, "drop_realtime", lambda: RecordingParams.events.append("drop_realtime"))
  monkeypatch.setattr(realtime, "set_core_affinity", lambda cores: RecordingParams.events.append("set_core_affinity"))

  def write_metadata(car_fingerprint):
    calls.append((threading.current_thread().name, car_fingerprint))
    started.set()
    assert release.wait(2.0)

  monkeypatch.setattr(longitudinal, "Params", RecordingParams)
  monkeypatch.setattr(longitudinal, "write_metadata", write_metadata)
  writer = longitudinal.HondaParamWriter()

  caller = threading.Thread(target=writer.put_many, args=(
    {
      "HondaGasAlphaParams": 0.2,
      "HondaGasFactorParams": 1.25,
      "HondaWindFactorParams": 0.9,
    },
    "HONDA_CLARITY",
  ))
  caller.start()
  caller.join(1.0)
  assert not caller.is_alive()
  assert started.wait(1.0)
  release.set()

  assert RecordingParams.instances[-1].values == {
    "HondaGasAlphaParams": 0.2,
    "HondaGasFactorParams": 1.25,
    "HondaWindFactorParams": 0.9,
  }
  assert calls == [("honda-param-writer", "HONDA_CLARITY")]
  assert RecordingParams.events[:2] == ["drop_realtime", "set_core_affinity"]
  assert RecordingParams.events[2:] == [
    ("put", "HondaGasAlphaParams", True),
    ("put", "HondaGasFactorParams", True),
    ("put", "HondaWindFactorParams", True),
  ]


def test_gas_alpha_load_is_fingerprint_safe_and_missing_value_is_independent(monkeypatch, tmp_path):
  metadata_path = tmp_path / "meta.json"
  metadata_path.write_text('{"car_fingerprint":"HONDA_CLARITY","learn_version":2}\n', encoding="utf-8")

  class LoadParams:
    value = None

    def get(self, key):
      assert key == "HondaGasAlphaParams"
      return self.value

  params = LoadParams()
  monkeypatch.setattr(longitudinal, "Params", lambda: params)
  monkeypatch.setattr(longitudinal, "LEARNER_META_PATH", str(metadata_path))

  assert longitudinal.load_gas_alpha("HONDA_CLARITY") == 0.0
  params.value = b"0.25"
  assert longitudinal.load_gas_alpha("HONDA_CLARITY") == 0.25
  assert longitudinal.load_gas_alpha("HONDA_CIVIC") == 0.0
  params.value = b"nan"
  assert longitudinal.load_gas_alpha("HONDA_CLARITY") == 0.0
  params.value = b"9"
  assert longitudinal.load_gas_alpha("HONDA_CLARITY") == 0.4
