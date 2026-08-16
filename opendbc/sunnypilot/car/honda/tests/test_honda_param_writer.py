import threading

from opendbc.sunnypilot.car.honda import longitudinal


class RecordingParams:
  instances = []

  def __init__(self):
    self.values = {}
    self.instances.append(self)

  def put(self, key, value):
    self.values[key] = value


def test_persistence_runs_off_thread(monkeypatch):
  started = threading.Event()
  release = threading.Event()
  calls = []

  def write_metadata(car_fingerprint):
    calls.append((threading.current_thread().name, car_fingerprint))
    started.set()
    assert release.wait(2.0)

  monkeypatch.setattr(longitudinal, "Params", RecordingParams)
  monkeypatch.setattr(longitudinal, "write_metadata", write_metadata)
  writer = longitudinal.HondaParamWriter()

  caller = threading.Thread(target=writer.put_many, args=(
    {"HondaGasFactorParams": 1.25},
    "HONDA_CLARITY",
  ))
  caller.start()
  caller.join(1.0)
  assert not caller.is_alive()
  assert started.wait(1.0)
  release.set()

  assert RecordingParams.instances[-1].values == {"HondaGasFactorParams": 1.25}
  assert calls == [("honda-param-writer", "HONDA_CLARITY")]
