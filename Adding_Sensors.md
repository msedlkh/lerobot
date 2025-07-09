# Adding Sensors to LeRobot

- This document provides a comprehensive guide for adding new sensors to the LeRobot system, based on the implementation patterns used for the NIRONE near-infrared spectrometer and HOPES tactile sensor.
- The assumption is that the sensor data is an n-dimensional array of data which can be stored as a numpy array. The stored data can then be streamed or recorded, depending on user preference, using the standard lerobot scripts.
- Note that data rate is currently capped at 30fps. See note on multi-rate sensors in [issue[#1174]](https://github.com/huggingface/lerobot/issues/1174)

## Overview

LeRobot supports a flexible sensor framework that allows integration of various sensor types through a standardized interface. Sensors are integrated into the robot system alongside cameras and motors, providing additional sensory input for data collection and robot control.

## General Changes 

These changes were added to enable the sensor class to be added to lerobot, to be noted if migrating to later versions of lerobot:
1. Added `sensors` folder for new sensors.
    - each new sensor should have entries added to `configs.py` and `utils.py`
    - each new sensor should have its own reader `mysensor.py` with the required helper functions (see below)
2. Added entries to `robots/configs.py`
3. Added `sensor_keys` to `dataset/lerobot_dataset.py`
4. Modified `get_features_from_robot` in `dataset/utils.py`
5. Added `run_sensor_capture` in `robot_devices/so100_remote.py`
6. Modified `_get_data` and `capture_observation` in `robot_devices/mobile_manipulator.py` (for leader)
7. Added entries for sensors in `robot_devices/manipulator.py` (for follower)

## Sensor Implementation Pattern

Based on the NIRONE and HOPES sensor implementations, adding a new sensor requires the following core components:

### 1. Sensor Configuration Class (`sensors/configs.py`)

Define a configuration dataclass that inherits from `SensorConfig` and registers with the choice registry:

```python
@SensorConfig.register_subclass("my_sensor")
@dataclass
class MySensorConfig(SensorConfig):
    """
    Configuration for My Sensor.
    
    Attributes:
        connection_param: Connection parameter for the sensor
        points: Number of data points the sensor returns
        mock: Whether to use mock data for testing
    """
    
    connection_param: str
    points: int = 64
    mock: bool = False

    def __post_init__(self):
        # Add validation logic here
        if self.points < 1:
            raise ValueError("Number of points must be at least 1.")
```

### 2. Sensor Implementation Class (`sensors/my_sensor.py`)

Create the main sensor class with standardized methods:

```python
class MySensor:
    def __init__(self, config: MySensorConfig):
        self.config = config
        self.connected = False
        self.mock = config.mock
        self.sensor_data = np.zeros(config.points, dtype=np.float32)
        
        # Threading support for async operations
        self.thread: Optional[threading.Thread] = None
        self.stop_event: Optional[threading.Event] = None
        self.logs = {}

    def connect(self):
        """Connect to the sensor device"""
        if self.connected:
            raise RobotDeviceAlreadyConnectedError("Sensor is already connected.")
        
        if self.mock:
            self.connected = True
            return
            
        # Implement actual connection logic here
        self.connected = True

    def disconnect(self):
        """Disconnect from the sensor device"""
        if not self.connected:
            raise RobotDeviceNotConnectedError("Sensor is not connected.")
        
        # Clean up threading
        if self.thread is not None:
            if self.stop_event is not None:
                self.stop_event.set()
            self.thread.join()
            self.thread = None
            self.stop_event = None

        # Implement disconnection logic here
        self.connected = False

    def read(self) -> np.ndarray:
        """Read data from the sensor synchronously"""
        if not self.connected:
            raise RobotDeviceNotConnectedError("Sensor is not connected.")
      
        if self.mock:
            return np.random.random(self.config.points).astype(np.float32)
        
        start_time = time.perf_counter()
        
        try:
            # Implement actual sensor reading logic here
            sensor_array = self._read_sensor_data()
            
            # Capture timing information
            self.logs["delta_timestamp_s"] = time.perf_counter() - start_time
            self.logs["timestamp_utc"] = capture_timestamp_utc()
            
            return sensor_array
            
        except Exception as e:
            raise RobotDeviceNotConnectedError(f"Failed to read sensor data: {e}") from e

    def async_read(self) -> np.ndarray:
        """Start/get asynchronous sensor reading"""
        if not self.connected:
            raise RobotDeviceNotConnectedError("Sensor is not connected.")
        
        if self.thread is None:
            self.stop_event = threading.Event()
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()

        num_tries = 0
        while True:
            if self.sensor_data is not None:
                return self.sensor_data
            if num_tries > 10:
                raise RobotDeviceNotConnectedError("Failed to read data from sensor.")
            busy_wait(0.1)
            num_tries += 1

    def _read_loop(self):
        """Internal method for continuous reading in a separate thread"""
        while self.stop_event is not None and not self.stop_event.is_set():
            try:
                self.sensor_data = self.read()
            except Exception as e:
                print(f"Error reading in thread: {e}")

    def _read_sensor_data(self) -> np.ndarray:
        """Implement the actual sensor-specific data reading logic"""
        # This is where you implement the sensor-specific communication
        pass
```

### 3. Sensor Factory Registration (`sensors/utils.py`)

Add your sensor to the factory functions:

```python
from lerobot.common.robot_devices.sensors.configs import (
    NIRONESensorConfig,
    HOPESSensorConfig,
    MySensorConfig,  # Add your sensor config
    SensorConfig,
)

def make_sensors_from_configs(sensor_configs: dict[str, SensorConfig]) -> list[Sensor]:
    sensors = {}

    for key, cfg in sensor_configs.items():
        if cfg.type == "nirone":
            from lerobot.common.robot_devices.sensors.nirone import NIRONESensor
            sensors[key] = NIRONESensor(cfg)
        elif cfg.type == "hopes":
            from lerobot.common.robot_devices.sensors.hopes import HOPESSensor
            sensors[key] = HOPESSensor(cfg)
        elif cfg.type == "my_sensor":  # Add your sensor
            from lerobot.common.robot_devices.sensors.my_sensor import MySensor
            sensors[key] = MySensor(cfg)
        else:
            raise ValueError(f"The sensor type '{cfg.type}' is not valid.")

    return sensors

def make_sensor(sensor_type, **kwargs) -> Sensor:
    if sensor_type == "nirone":
        from lerobot.common.robot_devices.sensors.nirone import NIRONESensor
        config = NIRONESensorConfig(**kwargs)
        return NIRONESensor(config)
    elif sensor_type == "hopes":
        from lerobot.common.robot_devices.sensors.hopes import HOPESSensor
        config = HOPESSensorConfig(**kwargs)
        return HOPESSensor(config)
    elif sensor_type == "my_sensor":  # Add your sensor
        from lerobot.common.robot_devices.sensors.my_sensor import MySensor
        config = MySensorConfig(**kwargs)
        return MySensor(config)
    else:
        raise ValueError(f"The sensor type '{sensor_type}' is not valid.")
```

### 4. Robot Configuration Integration (`robots/configs.py`)

Import and configure your sensor in robot configurations:

```python
from lerobot.common.robot_devices.sensors.configs import (
    HOPESSensorConfig,
    NIRONESensorConfig,
    MySensorConfig,  # Add your sensor config
    SensorConfig,
)

# In robot config classes, add sensor to the sensors field:
sensors: dict[str, SensorConfig] = field(
    default_factory=lambda: {
        "my_sensor": MySensorConfig(
            connection_param="/dev/ttyUSB0",
            points=64,
            mock=False,
        ),
        # ... other sensors
    }
)
```

## Implementation Checklist

When adding a new sensor, ensure you have implemented:

### Required Files
- [ ] `sensors/configs.py` - Add sensor configuration class
- [ ] `sensors/my_sensor.py` - Implement sensor class
- [ ] Update `sensors/utils.py` - Add factory registration
- [ ] Update `robots/configs.py` - Add imports and default configurations

### Required Methods
Your sensor class must implement:
- [ ] `__init__(self, config)` - Initialize with configuration
- [ ] `connect(self)` - Establish connection to sensor
- [ ] `disconnect(self)` - Clean up and disconnect
- [ ] `read(self) -> np.ndarray` - Synchronous data reading
- [ ] `async_read(self) -> np.ndarray` - Asynchronous data reading

### Required Properties
- [ ] `connected: bool` - Connection status
- [ ] `logs: dict` - Timing and metadata logs
- [ ] `mock: bool` - Mock mode support for testing

### Error Handling
- [ ] Raise `RobotDeviceAlreadyConnectedError` when connecting to already connected sensor
- [ ] Raise `RobotDeviceNotConnectedError` when operating on disconnected sensor
- [ ] Implement proper exception handling in read methods

### Threading Support
- [ ] Implement thread-safe asynchronous reading
- [ ] Proper cleanup of threads in disconnect method
- [ ] Use `threading.Event` for stop signaling

## Data Integration

Once implemented, sensor data will be automatically:

1. **Collected during observation capture** - Available as `observation.sensors.{sensor_name}`
2. **Included in dataset features** - Automatically added to robot features
3. **Logged with timing information** - Performance metrics tracked
4. **Handled in robot lifecycle** - Connected/disconnected with robot

## Testing Considerations

### Mock Mode
Always implement mock mode for testing without hardware:
```python
if self.mock:
    return np.random.random(self.config.points).astype(np.float32)
```

### Standalone Testing
Create a `__main__` section for standalone testing:
```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="My Sensor CLI")
    parser.add_argument("--connection_param", type=str, required=True)
    parser.add_argument("--mock", action="store_true", help="Use mock data")
    args = parser.parse_args()

    config = MySensorConfig(
        connection_param=args.connection_param,
        mock=args.mock,
    )

    sensor = MySensor(config)
    sensor.connect()
    print("Connected to sensor")
    
    # Test reading
    for i in range(5):
        data = sensor.read()
        print(f"Read {i+1}: {data}")
    
    sensor.disconnect()
```

## Examples

### NIRONE Near-Infrared Spectrometer
- **Connection**: Serial port communication (115200 baud)
- **Data**: Spectral measurements across configurable wavelength range
- **Features**: Wavelength vector configuration, averaging, light intensity control

### HOPES Tactile Sensor  
- **Connection**: USB HID device communication
- **Data**: Tactile pressure readings from sensor array
- **Features**: USB configuration, binary data parsing, taxel mapping

## Integration with Robot System

Sensors are automatically integrated into the robot system through:

1. **Initialization**: Sensors created from config during robot initialization
2. **Connection**: Connected when robot connects
3. **Data Collection**: Read during `capture_observation()` and `teleop_step()`
4. **Data Storage**: Stored as `observation.sensors.{name}` in datasets
5. **Disconnection**: Automatically disconnected when robot disconnects

The robot system handles all lifecycle management, error handling, and data formatting, allowing sensor implementations to focus on the sensor-specific communication logic.
