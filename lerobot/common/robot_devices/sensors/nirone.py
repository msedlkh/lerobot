import argparse
import struct
import threading
import time

import numpy as np
import serial
from tqdm import tqdm

from lerobot.common.robot_devices.sensors.configs import NIRONESensorConfig
from lerobot.common.robot_devices.utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceNotConnectedError,
    busy_wait,
)
from lerobot.common.utils.utils import capture_timestamp_utc


class NIRONESensor:
    def __init__(self, config: NIRONESensorConfig):
        self.connected = False
        self.ser: serial.Serial = serial.Serial(
            port=config.port,
            baudrate=115200,
            timeout=5,
            write_timeout=3,
        )
        self.min_wavelength = config.min_wavelength
        self.max_wavelength = config.max_wavelength
        self.points = min(config.points, 512)
        self.wavelength_vector = config.wavelength_vector
        if self.wavelength_vector is None:
            self.wavelength_vector = self._generate_wavelength_vector()
        self.averaging = config.averaging
        self.nir_array = np.zeros(self.points, dtype=np.float32)

        self.mock = config.mock

        self.thread = None
        self.stop_event: threading.Event | None = None
        self.logs = {}


    def connect(self):
        '''
        Connect to the NIRONE sensor.
        Raises:
            RobotDeviceAlreadyConnectedError: If the sensor is already connected.
            RobotDeviceNotConnectedError: If the sensor cannot be connected.
        '''

        if self.connected:
            raise RobotDeviceAlreadyConnectedError("NIRONE sensor is already connected.")
        if self.mock:
            # In mock mode, we do not need to open a serial connection
            self.connected = True
        else:
            if not self.ser.is_open:
                try:
                    self.ser.open()
                except serial.SerialException as e:
                    RobotDeviceNotConnectedError(f"Failed to open the serial port: {e}")
        

        self.connected = True

        if self.get_sensor_awake():
            # Set the wavelength vector to device limits 
            self.min_wavelength, self.max_wavelength = self.get_wavelength_range()
            self.wavelength_vector = self._generate_wavelength_vector()
            # Set the wavelength vector
            self.set_wavelength_vector()
            # Set the point scan averaging
            self.set_point_scan_avg(averaging=1, num_points=100)
            # Set the light mode to manual
            self.set_light_mode(mode=1)
            # Set the light intensity to 100%
            self.set_light_intensity(intensity=60)


    def disconnect(self):
        '''
        Disconnect from the NIRONE sensor.
        Raises:
            RobotDeviceNotConnectedError: If the sensor is not connected.
        '''

        if not self.connected:
            raise RobotDeviceNotConnectedError("NIRONE sensor is not connected.")
        
        if self.thread is not None:
            if self.stop_event is not None:
                self.stop_event.set()
            self.thread.join()
            self.thread = None
            self.stop_event = None

        # disconnect from the sensor
        self.ser.close()
        if self.ser.is_open:
            raise RobotDeviceNotConnectedError("Failed to close the serial port.")
        
        self.connected = False
        
    def __del__(self):
        '''
        Destructor to ensure the sensor is disconnected when the object is deleted.
        '''
        if getattr(self, 'connected', False):
            self.disconnect()

    def read(self) -> np.ndarray:
        '''
        Read data from the NIRONE sensor.
        Returns:
            np.ndarray: The measurement data as a NumPy array.
        Raises:
            RobotDeviceNotConnectedError: If the sensor is not connected or the serial port is not open.
        '''

        if not self.connected:
            raise RobotDeviceNotConnectedError("NIRONE sensor is not connected.")
        if self.mock:
            return np.zeros(self.points, dtype=np.float32)
        if not self.ser.is_open:
            raise RobotDeviceNotConnectedError("Serial port is not open.")
        
        start_time = time.perf_counter()
        # Set the sensor to perform a measurement scan
        self.set_measurement_scan()
        nir_array = self.get_measurement_scan()

        # Capture the time taken for the measurement
        self.logs["delta_timestamp_s"] = time.perf_counter() - start_time
        self.logs["timestamp_utc"] = capture_timestamp_utc()
        
        self.nir_array = nir_array

        return nir_array
    
    def read_loop(self):
        while self.stop_event is not None and not self.stop_event.is_set():
            try:
                self.nir_array = self.read()
            except Exception as e:
                print(f"Error reading in thread: {e}")
    
    def async_read(self):
        '''
        Start an asynchronous read from the NIRONE sensor.
        This method starts a new thread that continuously reads data from the sensor.
        Raises:
            RobotDeviceNotConnectedError: If the sensor is not connected or the serial port is not open.
        '''

        if not self.connected:
            raise RobotDeviceNotConnectedError("NIRONE sensor is not connected.")
        
        if self.thread is None:
            self.stop_event = threading.Event()
            self.thread = threading.Thread(target=self.read_loop, daemon=True)
            self.thread.start()

        num_tries = 0
        while True:
            if self.nir_array is not None:
                return self.nir_array
            if num_tries > 10:
                raise RobotDeviceNotConnectedError("Failed to read data from NIRONE sensor.")
            busy_wait(0.1)
            num_tries += 1


    def send_command(self, command: str):
        '''
        Send a command to the NIRONE sensor.
        Args:
            command (str): The command to send to the sensor.
        Raises:
            RobotDeviceNotConnectedError: If the sensor is not connected or the serial port is not open.
        '''

        if not self.connected:
            raise RobotDeviceNotConnectedError("NIRONE sensor is not connected.")
        if self.mock:
            return
        if not self.ser.is_open:
            raise RobotDeviceNotConnectedError("Serial port is not open.")
        self.ser.write(str.encode(command + "\n"))

    def read_response(self) -> str:
        '''
        Read the response from the NIRONE sensor.
        Returns:
            str: The response from the sensor.
        Raises:
            RobotDeviceNotConnectedError: If the sensor is not connected or the serial port is not open.
        '''

        if not self.connected:
            raise RobotDeviceNotConnectedError("NIRONE sensor is not connected.")
        if self.mock:
            return "Mock response"
        if not self.ser.is_open:
            raise RobotDeviceNotConnectedError("Serial port is not open.")
        while self.ser.in_waiting == 0:
            busy_wait(0.01)
        # Read the response from the sensor
        reply = self.ser.read(self.ser.in_waiting).decode('utf-8', errors='ignore')
        
        return reply.strip()
    
    def read_binary_response(self, total_bytes: int) -> list[float]:
        '''
        Read a binary response from the NIRONE sensor.
        Args:
            total_bytes (int): The total number of bytes to read.
        Returns:
            list[float]: The binary data as a list of floats.
        Raises:
            RobotDeviceNotConnectedError: If the sensor is not connected or the serial port is not open.
        '''

        if not self.connected:
            raise RobotDeviceNotConnectedError("NIRONE sensor is not connected.")
        if self.mock:
            return [0.0] * (total_bytes // 4)
        b_to_read = 0
        b_cumulatively_read = 0
        buffer = bytearray()
        # Loop until all bytes have been received
        while b_cumulatively_read < total_bytes:
            b_to_read = self.ser.in_waiting
            if b_to_read > 0:
                buffer.extend(self.ser.read(b_to_read))
                b_cumulatively_read += b_to_read
            busy_wait(0.01)
        # Convert the binary data into a float array
        return list(struct.unpack('f' * (total_bytes // 4), buffer))
    
    def send_and_read_command(self, command: str) -> str:
        '''
        Send a command to the NIRONE sensor and read the response.
        Args:
            command (str): The command to send to the sensor.
        Returns:
            str: The response from the sensor.
        Raises:
            RobotDeviceNotConnectedError: If the sensor is not connected or the serial port is not open.
        '''

        self.send_command(command)
        return self.read_response()

    def set_wavelength_vector(self) -> list[float]:
        '''
        Set the wavelength vector for the NIRONE sensor.
        Returns:
            list[float]: The wavelength vector that was set.
        Raises:
            RobotDeviceNotConnectedError: If the sensor is not connected or the serial port is not open.
        '''
        # use tqdm to show progress bar
        for i, wl in tqdm(enumerate(self.wavelength_vector), total=len(self.wavelength_vector), desc="Setting Wavelength Vector"):
            self.send_and_read_command(f"W{i},{wl}")

        return self.wavelength_vector
    
    def _generate_wavelength_vector(self) -> list[float]:
        '''
        Generate a wavelength vector based on the configuration.
        Returns:
            list[float]: The generated wavelength vector.
        Raises:
            ValueError: If the number of points is less than 2.
        '''
        return [
            self.min_wavelength + i * (self.max_wavelength - self.min_wavelength) / (self.points - 1)
            for i in range(self.points)
        ]

    def set_point_scan_avg(self, averaging: int = 1, num_points: int = 100):
        '''
        Set the point scan averaging for the NIRONE sensor.
        Args:
            averaging (int): The number of scans to average (default is 1).
            num_points (int): The number of points in the wavelength vector (default is 100).
        Returns:
            str: The response from the sensor.
        Raises:
            ValueError: If the averaging is less than 1 or if the number of points is less than 1.
            RobotDeviceNotConnectedError: If the sensor is not connected or the serial port is not open.
        '''
        return self.send_and_read_command(f"V{num_points},{averaging}")
    
    def set_light_mode(self, mode: int = 0):
        '''
        Set the light mode for the NIRONE sensor.
        Args:
            mode (int): The light mode to set (0 for manual, 1 for automatic).
        Returns:
            str: The response from the sensor.
        Raises:
            ValueError: If the mode is not 0 or 1.
            RobotDeviceNotConnectedError: If the sensor is not connected or the serial port is not open.
        '''
        if mode not in [0, 1]:
            raise ValueError("Mode must be 0 (manual) or 1 (automatic).")
        return self.send_and_read_command(f"LM{mode}")
    
    def set_light_intensity(self, intensity: int = 100):
        '''
        Set the light intensity for the NIRONE sensor.
        Args:
            intensity (int): The light intensity to set (0 to 100).
        Returns:
            str: The response from the sensor.
        Raises:
            ValueError: If the intensity is not between 0 and 100.
            RobotDeviceNotConnectedError: If the sensor is not connected or the serial port is not open.
        '''
        if not (0 <= intensity <= 100):
            raise ValueError("Intensity must be between 0 and 100.")
        return self.send_and_read_command(f"LI{intensity}")

    def set_measurement_scan(self):
        '''Set the sensor to perform a measurement scan.'''
        wait_time = self.get_measurement_time()/1000  # Convert milliseconds to seconds
        self.send_command("XM")
        busy_wait(wait_time)
        reply = self.read_response()
        if reply != "Measurement ready":
            raise RuntimeError(f"Unexpected response: {reply}")
        
    
    def get_sensor_awake(self) -> bool:
        '''
        Check if the sensor is awake and responsive.
        Returns:
            bool: True if the sensor is awake, False otherwise.
        '''
        if self.mock:
            return True
        reply = self.send_and_read_command("!")
        return reply != ""

    def get_measurement_scan(self) -> np.ndarray:
        '''
        Get the measurement scan data from the sensor.
        Returns:
            np.ndarray: The measurement data as a NumPy array.
        '''
        points = self.points
        if self.mock:
            return np.zeros(points, dtype=np.float32)
        
        self.send_command(f"Xm0,{points}")
        total_bytes = 4 * points
        # convert the response to a NumPy array
        nir_array = np.array(self.read_binary_response(total_bytes), dtype=np.float32)

        return nir_array


    def get_serial_number(self) -> str:
        '''
        Get the serial number of the NIRONE sensor.
        Returns:
            str: The serial number of the sensor.
        Raises:
            RobotDeviceNotConnectedError: If the sensor is not connected or the serial port is not open.
        '''
        return self.send_and_read_command("h2").strip()
    
    def get_wavelength_range(self) -> tuple[float, float]:
        '''
        Get the wavelength range of the NIRONE sensor.
        Returns:
            tuple[float, float]: A tuple containing the minimum and maximum wavelengths.
        Raises:
            RobotDeviceNotConnectedError: If the sensor is not connected or the serial port is not open.
        '''
        min_wl = float(self.send_and_read_command("h3").split(": ")[1])
        max_wl = float(self.send_and_read_command("h4").split(": ")[1])
        return max(min_wl, self.min_wavelength), min(max_wl, self.max_wavelength)
    
    def get_measurement_time(self) -> int:
        return int(self.send_and_read_command("E0").split(": ")[1])
    
    def get_sensor_temperature(self) -> float:
        if self.mock:
            return 25.0
        temp_str = self.send_and_read_command("St").split(": ")[1]
        try:
            return float(temp_str)
        except ValueError:
            raise RuntimeError(f"Failed to parse temperature: {temp_str}")
    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NIRONE Sensor CLI")
    parser.add_argument("--port", type=str, required=True, help="Serial port for the NIRONE sensor")
    parser.add_argument("--points", type=int, default=512, help="Number of points in the wavelength vector")
    args = parser.parse_args()

    config = NIRONESensorConfig(
        serial_number="NIRONE12345",
        port=args.port,
        points=args.points,
    )

    sensor = NIRONESensor(config)
    sensor.connect()
    print(f"Connected to NIRONE sensor with serial number: {sensor.get_serial_number()}")
    # read in a loop 10 times
    for _ in range(10):
        print(f"Reading data from sensor at {capture_timestamp_utc()}, temperature: {sensor.get_sensor_temperature()}°C")
        print("Setting measurement data...")
        sensor.set_measurement_scan()
        print(f"Wavelength vector: {sensor.get_measurement_scan()}")
    sensor.disconnect()