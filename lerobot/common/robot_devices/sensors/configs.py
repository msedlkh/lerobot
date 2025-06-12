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
        
