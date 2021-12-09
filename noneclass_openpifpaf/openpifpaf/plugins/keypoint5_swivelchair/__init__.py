import openpifpaf

from . import keypoint5_kp


def register():
    openpifpaf.DATAMODULES['keypoint5_swivelchair'] = keypoint5_kp.Keypoint5Kp