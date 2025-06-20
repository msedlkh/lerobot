import abc
from dataclasses import dataclass

import draccus

@dataclass
class SensorConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)
    
@SensorConfig.register_subclass("nirone")
@dataclass
class NIRONESensorConfig(SensorConfig):
    """
    Configuration for the NIRONE sensor.
    
    Attributes:
        port (str): Serial port to which the NIRONE device is connected.
        points (int): Number of points in the wavelength vector (max 512).
    """
    
    serial_number: str
    port: str
    points: int = 512
    min_wavelength: float = 1550.0
    max_wavelength: float = 1950.0
    wavelength_vector: list[float] = None
    averaging: int = 1
    mock: bool = False

    def __post_init__(self):
        if self.points > 512:
            raise ValueError("Maximum number of points is 512.")
        
        if not (self.min_wavelength < self.max_wavelength):
            raise ValueError("Minimum wavelength must be less than maximum wavelength.")
        
        if not isinstance(self.port, str) or not self.port:
            raise ValueError("Port must be a non-empty string.")
        if not (self.min_wavelength >= 0 and self.max_wavelength >= 0):
            raise ValueError("Wavelength values must be non-negative.")

@SensorConfig.register_subclass("hopes")
@dataclass
class HOPESSensorConfig(SensorConfig):
    """
    Configuration for the HOPES tactile sensor.

    Attributes:
        vendor_id (int): USB vendor ID for the HOPES device.
        product_id (int): USB product ID for the HOPES device.
        interface (int): USB interface number.
        endpoint (int): USB endpoint for interrupt input.
        b_packet_len (int): Length of the USB packet to read.
        usb_timeout (int): Timeout for USB operations in milliseconds.
        points (int): Number of points in the tactile sensor array.
        mock (bool): Whether to use a mock sensor instead of a real one.
    """
    
    vendor_id: int = 0x0000
    product_id: int = 0x00FF
    interface: int = 2
    endpoint: int = 0x84
    packet_len: int = 125
    timeout: int = 1000
    points: int = 65
    mock: bool = False

    def __post_init__(self):
        if self.points < 1:
            raise ValueError("Number of points must be at least 1.")
        if self.timeout < 1:
            raise ValueError("Timeout must be at least 1 millisecond.")
        if not (0 <= self.vendor_id <= 0xFFFF):
            raise ValueError(f"Vendor ID must be between 1 and 65535. Vendor ID is {self.vendor_id}.")
        if not (0 <= self.product_id <= 0xFFFF):
            raise ValueError(f"Product ID must be between 1 and 65535. Product ID is {self.product_id}.")
    
