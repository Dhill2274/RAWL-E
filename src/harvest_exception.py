class BerryException(Exception):
    def __init__(self):
        super().__init__

class FileExistsException(BerryException):
    def __init__(self, file_name):
        super().__init__
        self.file_name= file_name
    
    def __str__(self):
        return ("File {f} already exists".format(f=self.file_name))

class NoBerriesException(BerryException):
    def __init__(self, agent_id=None, coordinates=None):
        super().__init__
        self.agent_id = agent_id
        self.coordinates = coordinates
    
    def __str__(self):
        if self.coordinates == None:
            return ("Agent {a} has found no berries in the grid".format(a=self.agent_id))
        elif self.agent_id == None:
            return ("Coordinates {c} has no berries".format(c=self.coordinates))
        else:
            return ("Agent {a} trying to forage at {c} which has no berry".format(a=self.agent_id, c=self.coordinates))
        
class NumAgentsException(BerryException):
    def __init__(self, num_agents, num_expected_agents):
        super().__init__
        self.num_agents = num_agents
        self.num_expected_agents = num_expected_agents
    
    def __str__(self):
        return ("Expected {n} agents and got {m}".format(n=self.num_agents, m=self.num_expected_agents))

class OutOfBounds(BerryException):
    def __init__(self, agent_id, coordinates):
        super().__init__
        self.agent_id = agent_id
        self.coordinates = coordinates
    
    def __str__(self):
        return("Coordinates {c} are out of bounds for agent {a}".format(c=self.coordinates,a=self.agent_id))
    
class NoEmptyCells(BerryException):
    def __init__(self):
        pass

    def __str__(self):
        return ("No empty cells in grid")