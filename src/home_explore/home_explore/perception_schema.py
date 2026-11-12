"""明确检测框坐标格式，再统一转成 0～1 坐标。"""

import copy
import math
from .core import validate_observation


def decode_detection_response(raw, width, height):
    value = copy.deepcopy(raw)
    fmt = value.get('bbox_format')
    if fmt not in ('norm1000', 'norm1', 'pixels'):
        raise ValueError('模型必须声明bbox_format为norm1000、norm1或pixels')
    for detection in value.get('detections', []):
        box = detection.get('bbox')
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError('bbox必须有4个坐标')
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in box):
            raise ValueError('bbox坐标必须有限')
        scale = (
            [1000] * 4
            if fmt == 'norm1000'
            else ([width, height, width, height] if fmt == 'pixels' else [1] * 4)
        )
        detection['bbox'] = [v / s for v, s in zip(box, scale)]
    return validate_observation(value)
