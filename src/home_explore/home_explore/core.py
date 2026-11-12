"""任务与检测结果校验，以及通用角度计算。"""

from __future__ import annotations

import json
import math
import re

LANDMARKS = ('sofa', 'bed', 'dining_table', 'tea_table', 'bookshelf', 'cupboard', 'chair')


def json_object(text):
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text)
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError('模型没有返回合法 JSON') from exc
    if not isinstance(result, dict):
        raise ValueError('模型 JSON 必须是对象')
    return result


def validate_plan(data, allowed_targets=None):
    # 明确拒绝不支持的要求，避免悄悄丢失用户约束。
    if data.get('unsupported'):
        raise ValueError('当前任务不支持：' + str(data['unsupported']))
    steps = data.get('steps')
    if not isinstance(steps, list) or not 1 <= len(steps) <= 5:
        raise ValueError('任务必须包含 1 至 5 个地标步骤')
    out = []
    for step in steps:
        if not isinstance(step, dict) or step.get('target') not in (
            LANDMARKS if allowed_targets is None else allowed_targets
        ):
            raise ValueError('指令中的地标不在支持列表中')
        if step.get('relation', 'near') != 'near':
            raise ValueError('第一版仅支持到地标附近；不支持精确左右或背后关系')
        out.append(
            {
                'target': step['target'],
                'relation': 'near',
                'description': str(step.get('description', ''))[:200],
            }
        )
    return out


def validate_observation(data):
    detections = data.get('detections')
    if not isinstance(detections, list) or len(detections) > 20:
        raise ValueError('视觉结果缺少 detections 列表')
    valid = []
    for obj in detections:
        if not isinstance(obj, dict) or obj.get('label') not in LANDMARKS:
            raise ValueError('视觉结果包含不支持的类别')
        confidence = obj.get('confidence')
        bbox = obj.get('bbox')
        if isinstance(confidence, bool) or not isinstance(confidence, (float, int)):
            raise ValueError('confidence 必须是数值')
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('confidence 超出范围')
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError('bbox 必须有四个归一化坐标')
        if any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1
            for x in bbox
        ):
            raise ValueError('bbox 坐标超出范围')
        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise ValueError('bbox 范围无效')
        valid.append({'label': obj['label'], 'confidence': float(confidence), 'bbox': bbox})
    return {'detections': valid, 'description': str(data.get('description', ''))[:600]}


def angle_error(a, b):
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))
