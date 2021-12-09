import openpifpaf

from . import furniture_kp

def register():
    openpifpaf.DATAMODULES['furniture'] = furniture_kp.FurnitureKp
