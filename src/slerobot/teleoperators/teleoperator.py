import abc
from typing import Dict, Any


class Teleoperator(abc.ABC):
    """
    Abstract base class for teleoperator controllers.
    
    Defines the interface for controllers that can be either:
    - Quest 3s hand tracker (MQTT-based)
    - Visuomotor policy
    - Other teleoperation sources
    """
    
    @abc.abstractmethod
    def connect(self) -> None:
        """
        Establish connection with the controller.
        
        This method should initialize the connection to the teleoperation source
        (e.g., Quest 3s MQTT broker, policy server, etc.).
        
        Raises:
            ConnectionError: If connection fails
        """
        pass
    
    @abc.abstractmethod
    def disconnect(self) -> None:
        """
        Disconnect from the controller and perform any necessary cleanup.
        
        This method should clean up resources such as MQTT connections,
        thread cleanup, etc.
        """
        pass
    
    @abc.abstractmethod
    def get_action(self) -> Dict[str, Any]:
        """
        Retrieve the latest action from the controller.
        
        This method should return the most recent action data from the teleoperation source.
        If no action is available, return a default/neutral action.
        
        Returns:
            RobotAction: A dictionary representing the teleoperator's current actions.
                Expected keys may include:
                - 'timestamp': datetime of action capture
                - 'position': dict with x, y, z coordinates
                - 'rotation': dict with euler angles (w, p, r)
                - 'buttons': dict with button states (trigger, grip, etc.)
        """
        pass 