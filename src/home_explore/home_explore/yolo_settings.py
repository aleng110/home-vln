"""四类家具本地检测器的配置读取与校验。"""

from pathlib import Path
import math
import yaml

TARGETS = ('sofa', 'tea_table', 'dining_table', 'chair')


def load_settings(path):
    data = yaml.safe_load(Path(path).read_text())
    prompts, labels = data['text_prompts'], data['canonical_labels']
    if len(prompts) != len(labels) or len(set(labels)) != len(labels) or set(labels) != set(TARGETS):
        raise ValueError('YOLO配置必须一一对应sofa、tea_table、dining_table、chair四类')
    thresholds = data['minimum_confidence']
    for label in labels:
        value = thresholds[label]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 < value < 1
        ):
            raise ValueError('YOLO类别阈值必须在0与1之间')
    return dict(
        prompts=prompts,
        labels=labels,
        thresholds=thresholds,
        model=data['models']['open_vocabulary'],
        image_size=int(data.get('image_size', 640)),
        device=str(data.get('device', '0')),
    )
