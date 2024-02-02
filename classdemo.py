class Inner():
    def __init__(self):
        self.atr2 = dict()
        self.WhatTheHeck = True
        
    def HaveFun(self,msg):
        print(msg) #screenoutput
        return "the real function output"
        
        
class Outer():
    def __init__(self):
        self.atr1 = Inner()
        
        
print("nothing happend yet")

objO = Outer()
objO.WhatTheHeck = False


print("learning")

print(type(objO))
print(type(objO.atr1))
print(type(objO.atr1.atr2))
objO.atr1.atr2["key"] = 123
print(objO.atr1.atr2["key"])

print("playtime")
#print(objO.atr1.atr2.atr3["key"])
#print(objO.atr1["key"])

print(objO.atr1.HaveFun("yes please"))
