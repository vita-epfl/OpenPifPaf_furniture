import openpifpaf

from . import pascal3d_kp

def register():
    openpifpaf.DATAMODULES['pascal3d'] = pascal3d_kp.Pascal3dKp
