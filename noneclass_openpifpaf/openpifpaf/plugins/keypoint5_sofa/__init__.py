import openpifpaf

from . import keypoint5_kp


def register():
    openpifpaf.DATAMODULES['keypoint5_sofa'] = keypoint5_kp.Keypoint5Kp