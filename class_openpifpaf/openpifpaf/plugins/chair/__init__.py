import openpifpaf

from . import pascal3d_kp

def register():
    openpifpaf.DATAMODULES['chair'] = pascal3d_kp.Pascal3dKp
