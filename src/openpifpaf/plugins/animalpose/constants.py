
import os

import numpy as np

_CATEGORIES = ['limit 120', 'no overtaking', 'no overtaking by lorries', 'crossing with a road without priority ahead', 'principal road with prioritiy',
              'yield sign', 'stop', 'generall prohibition of vehicular traffic in both directions', 'Prohibition of large goods vehicles', 'no entry',
              'other danger', 'bend to left', 'bend to right', 'double bend', 'uneven road', 'slippery road', 'road narrows on right', 'road works',
              'traffic lights', 'pedestrian crossing ahead', 'Presence of children', 'bicycle traffic/crossing', 'snow', 'Wild animals crossing',
              'free drive', 'must turn right ahead', 'must turn left ahead', 'must continue straight ahead', 'must continue straight ahead or turn right',
              'must continue straight ahead or turn left', 'circumvent the obstacle on the right side', 'circumvent the obstacle on the left side',
              'roundabout', 'end of no overtaking restriction', 'end of no overtaking by lorries']  # only for preprocessing



SIGN_KEYPOINTS = [
    'Corner_inf_left',         # 1
    'Corner_inf_right',        # 2
    'Corner_sup_right',        # 3
    'Corner_sup_left',         # 4
]


ALTERNATIVE_NAMES = [
    'Corner_inf_left',         # 1
    'Corner_inf_right',        # 2
    'Corner_sup_right',        # 3
    'Corner_sup_left',         # 4
]


SIGN_SKELETON = [
    (1, 2), (2, 3), (3,4), (4, 1)  #Initialize the outlines of the sign
]


SIGN_SIGMAS = [         #Valueà 0.09 pour tester, à ajuster par la suite
    0.09,               # Corner_inf_left
    0.09,               # Corner_inf_right
    0.09,               # Corner_sup_right
    0.09,               # Corner_sup_left
]

split, error = divmod(len(SIGN_KEYPOINTS), 4)
SIGN_SCORE_WEIGHTS = [1 * error, 1 * error, 1 * error, 1 * error]    

SIGN_CATEGORIES = ['traffic sign']

SIGN_POSE = np.array([         
    [0.0, 0.0, 1],      # 'Corner_inf_left'        
    [40.0, 0.0, 1],     # 'Corner_inf_right'      
    [40.0, 40.0, 1],    # 'Corner_sup_right'      
    [0.0, 40.0, 1],     # 'Corner_sup_left'       
])


assert len(SIGN_POSE) == len(SIGN_KEYPOINTS) == len(ALTERNATIVE_NAMES) == len(SIGN_SIGMAS) \
       == len(SIGN_SCORE_WEIGHTS), "dimensions!"


def draw_ann(ann, *, keypoint_painter, filename=None, margin=0.5, aspect=None, **kwargs):
    from openpifpaf import show  # pylint: disable=import-outside-toplevel

    bbox = ann.bbox()
    xlim = bbox[0] - margin, bbox[0] + bbox[2] + margin
    ylim = bbox[1] - margin, bbox[1] + bbox[3] + margin
    if aspect == 'equal':
        fig_w = 5.0
    else:
        fig_w = 5.0 / (ylim[1] - ylim[0]) * (xlim[1] - xlim[0])

    with show.canvas(filename, figsize=(fig_w, 5), nomargin=True, **kwargs) as ax:
        ax.set_axis_off()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        if aspect is not None:
            ax.set_aspect(aspect)

        keypoint_painter.annotation(ax, ann)


def draw_skeletons(pose):
    from openpifpaf.annotation import Annotation  # pylint: disable=import-outside-toplevel
    from openpifpaf import show  # pylint: disable=import-outside-toplevel

    scale = np.sqrt(
        (np.max(pose[:, 0]) - np.min(pose[:, 0]))
        * (np.max(pose[:, 1]) - np.min(pose[:, 1]))
    )

    show.KeypointPainter.show_joint_scales = True
    show.KeypointPainter.font_size = 0
    keypoint_painter = show.KeypointPainter()

    ann = Annotation(
        keypoints=SIGN_KEYPOINTS, skeleton=SIGN_SKELETON, score_weights=SIGN_SCORE_WEIGHTS)
    ann.set(pose, np.array(SIGN_SIGMAS) * scale)
    os.makedirs('all-images', exist_ok=True)
    draw_ann(ann, filename='all-images/skeleton_animal.png', keypoint_painter=keypoint_painter)


def print_associations():
    for j1, j2 in SIGN_SKELETON:
        print(SIGN_SKELETON[j1 - 1], '-', SIGN_KEYPOINTS[j2 - 1])


if __name__ == '__main__':
    print_associations()
    draw_skeletons(SIGN_POSE)
