# hopes.py - A Python module for reading tactile sensor data from a USB device
# This module provides functionality to read and process data from a tactile sensor connected via USB.
# To read usb via usbhid-dump, use the command:
#   usbhid-dump -d 0x0000:0x00FF -e s

import argparse
import struct
import threading
import time
from typing import Optional, Tuple

import numpy as np
import usb.core
import usb.util

from lerobot.common.robot_devices.sensors.configs import HOPESSensorConfig
from lerobot.common.robot_devices.utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceNotConnectedError,
    busy_wait,
)
from lerobot.common.utils.utils import capture_timestamp_utc


class HOPESSensor:
    def __init__(self, config: HOPESSensorConfig):
        self.config = config
        self.dev = None
        self.connected = False
        
        # Data storage
        self.points = config.points
        self.recorded_nodes_adc = np.zeros([config.points, 1])
        self.recorded_nodes_pol = np.zeros([config.points, 1])
        self.tactile_data = np.ones(config.points, dtype=np.float32)
        
        # Threading support
        self.thread: Optional[threading.Thread] = None
        self.stop_event: Optional[threading.Event] = None
        self.logs = {}
        
        self.mock = config.mock

    def connect(self):
        """
        Connect to the tactile sensor.
        Raises:
            RobotDeviceAlreadyConnectedError: If the sensor is already connected.
            RobotDeviceNotConnectedError: If the sensor cannot be connected.
        """
        if self.connected:
            raise RobotDeviceAlreadyConnectedError("Tactile sensor is already connected.")
        
        if self.mock:
            self.connected = True
            return
        
        try:
            self.dev = self._init_usb_device()
            self.connected = True
            print(f"Connected to tactile sensor: {self.read()}")
        except Exception as e:
            raise RobotDeviceNotConnectedError(f"Failed to connect to tactile sensor: {e}") from e

    def disconnect(self):
        """
        Disconnect from the tactile sensor.
        Raises:
            RobotDeviceNotConnectedError: If the sensor is not connected.
        """
        if not self.connected:
            raise RobotDeviceNotConnectedError("Tactile sensor is not connected.")
        
        if self.thread is not None:
            if self.stop_event is not None:
                self.stop_event.set()
            self.thread.join()
            self.thread = None
            self.stop_event = None

        if not self.mock and self.dev is not None:
            self._cleanup_usb()
        
        self.connected = False

    def __del__(self):
        """Destructor to ensure the sensor is disconnected when the object is deleted."""
        if getattr(self, 'connected', False):
            self.disconnect()

    def read(self) -> np.ndarray:
        """
        Read data from the tactile sensor.
        Returns:
            np.ndarray: The tactile sensor data as a NumPy array.
        Raises:
            RobotDeviceNotConnectedError: If the sensor is not connected.
        """
        if not self.connected:
            raise RobotDeviceNotConnectedError("Tactile sensor is not connected.")
      
        if self.mock:
            return np.random.random(self.config.points).astype(np.float32)
        
        start_time = time.perf_counter()
        
        try:
            taxels, adc_vals, polarities, flag = self._read_sensor_data()
            
            # Convert to full array format
            tactile_array = np.ones(self.config.points, dtype=np.float32)
            if taxels is not None and len(taxels) > 0:
                tactile_array[taxels] = adc_vals
            
            self.tactile_data = tactile_array
            
            # Capture timing information
            self.logs["delta_timestamp_s"] = time.perf_counter() - start_time
            self.logs["timestamp_utc"] = capture_timestamp_utc()
            
            return tactile_array
            
        except Exception as e:
            raise RobotDeviceNotConnectedError(f"Failed to read tactile sensor data: {e}") from e

    def async_read(self) -> np.ndarray:
        """
        Start an asynchronous read from the tactile sensor.
        Returns:
            np.ndarray: The latest tactile sensor data.
        Raises:
            RobotDeviceNotConnectedError: If the sensor is not connected.
        """
        if not self.connected:
            raise RobotDeviceNotConnectedError("Tactile sensor is not connected.")
        
        if self.thread is None:
            self.stop_event = threading.Event()
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()

        num_tries = 0
        while True:
            if self.tactile_data is not None:
                return self.tactile_data
            if num_tries > 10:
                raise RobotDeviceNotConnectedError("Failed to read data from tactile sensor.")
            busy_wait(0.1)
            num_tries += 1

    def _read_loop(self):
        """Internal method for continuous reading in a separate thread."""
        while self.stop_event is not None and not self.stop_event.is_set():
            try:
                self.tactile_data = self.read()
            except Exception as e:
                print(f"Error reading in thread: {e}")

    def _init_usb_device(self):
        """Initialize USB device connection"""
        dev = usb.core.find(idVendor=self.config.vendor_id, idProduct=self.config.product_id)
        if dev is None:
            raise ValueError('USB device not found')
        
        # Detach kernel driver if active
        if dev.is_kernel_driver_active(self.config.interface):
            try:
                dev.detach_kernel_driver(self.config.interface)
            except usb.core.USBError as e:
                raise Exception(f"Could not detach kernel driver: {e}") from e
        
        return dev

    def _cleanup_usb(self):
        """Clean up USB resources"""
        try:
            if self.dev is not None:
                self.dev.attach_kernel_driver(self.config.interface)
        except Exception:
            pass

    def _read_sensor_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[int]]:
        """Read and parse sensor data from USB device"""
        try:
            # Read USB packet
            ret = self.dev.read(self.config.endpoint, self.config.packet_len, self.config.timeout)
            
            if len(ret) > 0:
                return self._parse_sensor_data(ret)
            else:
                return None, None, None, None
                
        except usb.core.USBTimeoutError:
            return None, None, None, None
        except Exception as e:
            print(f"USB read error: {e}")
            return None, None, None, None

    def _parse_sensor_data(self, raw_data) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Parse raw USB data into tactile sensor values"""
        _recorded_nodes_adc = np.zeros(self.config.points)
        
        # Unpack HID report data
        adc_data, xyzN = self._report_unpack(raw_data)  # noqa: N806
        g_flag = int.from_bytes(xyzN[3], "big")
        
        # Fill ADC array (skip index 0)
        _recorded_nodes_adc[1:len(adc_data)+1] = np.array(adc_data)
        
        # Remove saturated values (max 14-bit = 16383)
        _recorded_nodes_adc = np.where(_recorded_nodes_adc < 16383, _recorded_nodes_adc, 0)
        
        # Calculate polarity (change detection)
        pol_array = np.where(self.recorded_nodes_adc < _recorded_nodes_adc, -1, 1)
        self.recorded_nodes_adc = _recorded_nodes_adc
        self.recorded_nodes_pol = pol_array
        
        # Extract triggered taxels
        taxel_indices = np.where(self.recorded_nodes_adc)[0]
        adc_values = self.recorded_nodes_adc[taxel_indices].astype(int)
        pol_values = self.recorded_nodes_pol[taxel_indices].astype(int)
        
        return taxel_indices, adc_values, pol_values, g_flag

    def _report_unpack(self, report):
        """Unpack HID report data"""
        unpacked_data = struct.unpack('112s3fc', report)  # 112charstring, 3float, 1char
        adc_packed = unpacked_data[0]
        
        first_cnt = 0
        second_cnt = 0
        adc_unpacked = [0] * 64
        
        # 64 channels / 4 
        for _ in range(16):
            adc_unpacked[first_cnt + 0] = (adc_packed[second_cnt + 0] & 0xFF) << 6  | (adc_packed[second_cnt + 1] & 0xFC) >> 2
            adc_unpacked[first_cnt + 1] = (adc_packed[second_cnt + 1] & 0x03) << 12 | (adc_packed[second_cnt + 2] & 0xFF) << 4 | (adc_packed[second_cnt + 3] & 0xF0) >> 4
            adc_unpacked[first_cnt + 2] = (adc_packed[second_cnt + 3] & 0x0F) << 10 | (adc_packed[second_cnt + 4] & 0xFF) << 2 | (adc_packed[second_cnt + 5] & 0xC0) >> 6
            adc_unpacked[first_cnt + 3] = (adc_packed[second_cnt + 5] & 0x3F) << 8  | (adc_packed[second_cnt + 6] & 0xFF)
            
            first_cnt += 4
            second_cnt += 7
        
        return adc_unpacked, unpacked_data[1:5]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tactile Sensor CLI")
    parser.add_argument("--vendor_id", type=int, default=0x0000, help="USB Vendor ID")
    parser.add_argument("--product_id", type=int, default=0x00FF, help="USB Product ID")
    parser.add_argument("--points", type=int, default=65, help="Number of tactile sensor points")
    parser.add_argument("--mock", action="store_true", help="Use mock data")
    args = parser.parse_args()

    config = HOPESSensorConfig(
        vendor_id=args.vendor_id,
        product_id=args.product_id,
        points=args.points,
        mock=args.mock,
    )

    sensor = HOPESSensor(config)
    sensor.connect()
    print("Connected to tactile sensor")
    
    # Test synchronous reading
    for i in range(5):
        data = sensor.read()
        print(f"Read {i+1}: {np.sum(data)} total activation")
    
    # Test asynchronous reading
    print("Starting asynchronous read...")
    sensor.async_read()
    time_wait = time.time()
    while time.time() - time_wait < 10:
        data = sensor.tactile_data
        if data is not None:
            print(f"Data: {data}, Shape: {data.shape}, Sum: {np.sum(data)}")
    print(f"Sensor logs: {sensor.logs}")
    
    sensor.disconnect()