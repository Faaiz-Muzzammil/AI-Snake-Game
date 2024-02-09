from enum import Enum

class Color:
    RED = 1
    GREEN = 2
    WHITE = 3
    
    def __init__(self, code, name):
        self.code = code
        self.name = name
        
    def display_info(self):
        print(f"Color: {self.name}, Code: {self.code}")

# Creating an instance of the Color enumeration
red_color = Color.RED

# Calling the display method
red_color.display_info()
