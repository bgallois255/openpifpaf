
import openpifpaf

from . import traffic_sign_kp


def register():
    openpifpaf.DATAMODULES['traffic_sign'] = traffic_sign_kp.TrafficSignKp
