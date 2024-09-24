class Model():
    def __init__(self) -> None:
        pass
    
    def roomTemperatureSetting(self, roomTemperature=None, targetTemperature=None, mode="Cooling"):
        if mode == "Cooling":
            startTime, acTemperature, energyConsumption = self.function1(roomTemperature, targetTemperature)
        if mode == "heating":
            pass
        return startTime, acTemperature, energyConsumption
    
    def function1(self, roomTemperature, targetTemperature):
        acTemperature = 0
        startTime = 0
        energyConsumption = 0
        return startTime, acTemperature, energyConsumption


m = Model()
startTime, acTemperature, energyConsumption = m.roomTemperatureSetting(30, 26)
print("start at", startTime, "hours earlier with ac temperature set to", acTemperature, "degree, cost", energyConsumption, "watt")