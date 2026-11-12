import base64
import io
import math
from pathlib import Path
import queue
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import cv2
import numpy as np
from home_explore.core import validate_plan
from home_explore.semantic import Memory, surface_geometry, project_observed_point
from home_explore.yolo_client import YoloClient


class MaskGeometryTests(unittest.TestCase):
    def inputs(self):
        depth = np.ones((80, 80), np.float32)
        depth[20:60, 20:60] = 2.0
        mask = np.zeros((80, 80), bool)
        mask[20:60, 20:60] = True
        return (
            depth,
            mask,
            [80, 0, 40, 0, 80, 40, 0, 0, 1],
            [0, 0, 1],
            [0, math.sin(math.pi / 4), 0, math.cos(math.pi / 4)],
        )

    def test_mask_excludes_nearer_background_inside_box(self):
        depth, mask, k, t, q = self.inputs()
        masked = surface_geometry([0, 0, 1, 1], depth, k, t, q, mask)
        boxed = surface_geometry([0, 0, 1, 1], depth, k, t, q)
        self.assertAlmostEqual(masked['point'][0], 2.0, places=5)
        self.assertAlmostEqual(boxed['point'][0], 1.0, places=5)
        self.assertEqual(masked['method'], 'mask_front_depth_band')
        self.assertTrue(masked['partial'])

    def test_empty_mask_cannot_fall_back_to_bbox(self):
        depth, mask, k, t, q = self.inputs()
        self.assertIsNone(surface_geometry([0, 0, 1, 1], depth, k, t, q, np.zeros_like(mask)))

    def test_mask_and_depth_alignment_is_required(self):
        depth, mask, k, t, q = self.inputs()
        with self.assertRaises(ValueError):
            surface_geometry([0, 0, 1, 1], depth, k, t, q, mask[:40])

    def test_historical_surface_projection_uses_inverse_camera_transform(self):
        depth, mask, k, t, q = self.inputs()
        uv = project_observed_point([2, 0, 1], k, t, q, depth.shape)
        np.testing.assert_allclose(uv, [0.5, 0.5], atol=1e-8)
        self.assertIsNone(project_observed_point([-2, 0, 1], k, t, q, depth.shape))


class LocalEvidenceTests(unittest.TestCase):
    def geometry(self):
        return dict(
            point=[2.0, 1.0, 0.7],
            extent=[0.4, 0.3, 0.2],
            points=[],
            valid_points=80,
            spread=0.15,
            partial=True,
            method='mask_front_depth_band',
        )

    def test_yolo_score_uses_own_threshold_and_mask_evidence(self):
        m = Memory({'sofa': 0.35})
        m.add('sofa', 0.4, self.geometry(), [0, 0, 1], 10.0, 'a', source='yolo')
        o = m.add('sofa', 0.4, self.geometry(), [0.2, 0, 1], 12.0, 'b', source='yolo')
        self.assertTrue(m.quality(o, 12.0)['reliable'])

    def test_tracker_and_low_score_cannot_promote_target(self):
        m = Memory({'sofa': 0.35})
        m.add('sofa', 0.3, self.geometry(), [0, 0, 1], 10.0, 'a', source='yolo')
        o = m.add('sofa', 0.99, self.geometry(), [0.2, 0, 1], 12.0, 'b', source='tracker')
        self.assertFalse(m.quality(o, 12.0)['reliable'])

    def test_yolo_bbox_only_evidence_is_not_accepted(self):
        m = Memory({'sofa': 0.35})
        g = self.geometry()
        g['method'] = 'bbox_front_depth_band'
        m.add('sofa', 0.95, g, [0, 0, 1], 10.0, 'a', source='yolo')
        o = m.add('sofa', 0.95, g, [0.2, 0, 1], 12.0, 'b', source='yolo')
        self.assertFalse(m.quality(o, 12.0)['reliable'])

    def test_rejected_semantic_instance_is_not_selected_again(self):
        m = Memory({'sofa': 0.35})
        o = m.add('sofa', 0.9, self.geometry(), [0, 0, 1], 10.0, 'a', source='yolo')
        m.reject(o['id'], 'actually a wall', 10.0)
        self.assertIsNone(m.target('sofa'))
        m.clear_rejections()
        self.assertEqual(m.target('sofa')['id'], o['id'])

    def test_chair_supported_bed_rejected_in_four_class_task(self):
        allowed = ('sofa', 'tea_table', 'dining_table', 'chair')
        self.assertEqual(validate_plan({'steps': [{'target': 'chair'}]}, allowed)[0]['target'], 'chair')
        with self.assertRaises(ValueError):
            validate_plan({'steps': [{'target': 'bed'}]}, allowed)


class ProtocolTests(unittest.TestCase):
    def encoded(self, image, kind='.png'):
        ok, b = cv2.imencode(kind, image)
        self.assertTrue(ok)
        return base64.b64encode(b).decode('ascii')

    def client(self):
        client = YoloClient.__new__(YoloClient)
        client.settings = {
            'labels': ['sofa', 'tea_table', 'dining_table', 'chair'],
            'thresholds': {'sofa': 0.35, 'tea_table': 0.4, 'dining_table': 0.45, 'chair': 0.5},
        }
        return client

    def value(self):
        return dict(
            type='result',
            id='new',
            width=12,
            height=10,
            model='yoloe-26m-seg.pt',
            inference_ms=15.0,
            detections=[
                dict(
                    label='sofa',
                    confidence=0.6,
                    bbox=[0.1, 0.1, 0.9, 0.9],
                    mask_png=self.encoded(np.ones((10, 12), np.uint8) * 255),
                )
            ],
            overlay_jpeg=self.encoded(np.zeros((10, 12, 3), np.uint8), '.jpg'),
        )

    def test_roundtrip_keeps_binary_original_resolution_mask(self):
        d = self.client().decode(self.value(), (10, 12))
        self.assertEqual(d['masks'][0].shape, (10, 12))
        self.assertEqual(d['masks'][0].dtype, np.bool_)
        self.assertEqual(d['source'], 'yolo')

    def test_invalid_mask_cannot_reach_geometry(self):
        value = self.value()
        value['detections'][0]['mask_png'] = self.encoded(np.ones((5, 6), np.uint8) * 255)
        with self.assertRaises(ValueError):
            self.client().decode(value, (10, 12))

    def test_out_of_scope_class_is_rejected(self):
        value = self.value()
        value['detections'][0]['label'] = 'bed'
        with self.assertRaises(ValueError):
            self.client().decode(value, (10, 12))

    def test_cancelled_request_result_cannot_be_used_by_new_frame(self):
        client = self.client()
        client.closed = False
        client.lock = threading.Lock()
        client.ready = threading.Event()
        client.ready.set()
        client.ready_info = {'type': 'ready'}
        client.process = SimpleNamespace(poll=lambda: None, stdin=io.StringIO())
        client.responses = queue.Queue()
        client.responses.put(dict(type='result', id='cancelled_old_frame'))
        client.responses.put(self.value())
        with patch('home_explore.yolo_client.uuid.uuid4', return_value=SimpleNamespace(hex='new')):
            result = client.infer(np.zeros((10, 12, 3), np.uint8), lambda: None, timeout=1.0)
        self.assertEqual(result['detections'][0]['label'], 'sofa')


if __name__ == '__main__':
    unittest.main()
