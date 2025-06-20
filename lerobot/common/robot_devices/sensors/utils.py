from typing import Protocol

import numpy as np

from lerobot.common.robot_devices.sensors.configs import (
    NIRONESensorConfig,
    HOPESSensorConfig,
    SensorConfig,
)


# Defines a sensor type
class Sensor(Protocol):
    def connect(self): ...
    def read(self) -> np.ndarray: ...
    def async_read(self) -> np.ndarray: ...
    def disconnect(self): ...
    
def make_sensors_from_configs(sensor_configs: dict[str, SensorConfig]) -> list[Sensor]:
    sensors = {}

    for key, cfg in sensor_configs.items():
        if cfg.type == "nirone":
            from lerobot.common.robot_devices.sensors.nirone import NIRONESensor
            sensors[key] = NIRONESensor(cfg)
        elif cfg.type == "hopes":
            from lerobot.common.robot_devices.sensors.hopes import HOPESSensor
            sensors[key] = HOPESSensor(cfg)
        else:
            raise ValueError(f"The sensor type '{cfg.type}' is not valid.")

    return sensors

def make_sensor(sensor_type, **kwargs) -> Sensor:
    if sensor_type == "nirone":
        from lerobot.common.robot_devices.sensors.nirone import NIRONESensor

        config = NIRONESensorConfig(**kwargs)
        return NIRONESensor(config)

    else:
        raise ValueError(f"The sensor type '{sensor_type}' is not valid.")