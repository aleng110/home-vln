"""ROS 端通过 JSONL 调用独立 Conda 分割进程。"""

import base64
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import uuid
import cv2
import numpy as np
from .core import validate_observation
from .yolo_settings import load_settings


class YoloClient:
    def __init__(self, project_root, log_path):
        root = Path(project_root)
        self.settings = load_settings(root / 'config/yolo_scene_classes.yaml')
        self.responses = queue.Queue(maxsize=8)
        self.ready = threading.Event()
        self.ready_info = None
        self.closed = False
        self.lock = threading.Lock()
        self.log = Path(log_path).open('a', buffering=1)
        # 工作进程只继承必要环境，不传入云端凭据或 ROS 库路径。
        env = {
            k: os.environ[k]
            for k in ('HOME', 'LANG', 'LC_ALL', 'CUDA_VISIBLE_DEVICES', 'YOLO_PYTHON')
            if k in os.environ
        }
        env.update(PATH='/usr/local/bin:/usr/bin:/bin', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
        self.process = subprocess.Popen(
            [
                'bash',
                str(root / 'scripts/yolo.sh'),
                'python',
                str(Path(__file__).resolve().with_name('yolo_worker.py')),
                '--project-root',
                str(root),
                '--parent-pid',
                str(os.getpid()),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=True,
        )
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                if len(line) > 16_000_000:
                    raise ValueError('YOLO响应过大')
                value = json.loads(line)
                if value.get('type') == 'ready':
                    if set(value.get('labels', [])) != set(self.settings['labels']):
                        raise ValueError('YOLO启动类别与配置不符')
                    self.ready_info = value
                    self.ready.set()
                else:
                    self._put(value)
        except Exception as exc:
            self._put(dict(type='transport_error', message=str(exc)))
        finally:
            self._put(dict(type='eof', message='YOLO工作进程退出；查看工作进程日志'))
            self.ready.set()

    def _put(self, value):
        try:
            self.responses.put_nowait(value)
        except queue.Full:
            try:
                self.responses.get_nowait()
            except queue.Empty:
                pass
            self.responses.put_nowait(value)

    def wait_ready(self, check, timeout=60.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            check()
            if self.ready.wait(0.1):
                if self.ready_info is None:
                    raise RuntimeError('YOLO模型未能启动；查看yolo_worker日志')
                return self.ready_info
        raise TimeoutError('等待YOLO模型初始化超时')

    def infer(self, image, check, timeout=12.0):
        self.wait_ready(check)
        with self.lock:
            check()
            if self.closed or self.process.poll() is not None:
                raise RuntimeError('YOLO工作进程不可用')
            ok, encoded = cv2.imencode('.png', image)
            if not ok:
                raise ValueError('RGB编码失败')
            request_id = uuid.uuid4().hex
            self.process.stdin.write(
                json.dumps(dict(id=request_id, image_png=base64.b64encode(encoded).decode('ascii'))) + '\n'
            )
            self.process.stdin.flush()
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                check()
                try:
                    value = self.responses.get(timeout=0.1)
                except queue.Empty:
                    continue
                if value.get('type') in ('eof', 'transport_error'):
                    raise RuntimeError(value['message'])
                if value.get('id') != request_id:
                    continue  # 丢弃取消任务留下的迟到响应。
                if value.get('type') == 'error':
                    raise ValueError(value['message'])
                return self.decode(value, image.shape[:2])
        raise TimeoutError('YOLO推理响应超时')

    def decode(self, value, shape):
        height, width = shape
        if value.get('type') != 'result' or (value.get('height'), value.get('width')) != shape:
            raise ValueError('YOLO响应图像尺寸或类型不匹配')
        raw = value.get('detections')
        clean = validate_observation(dict(detections=raw, description='本地实例分割'))
        masks = []
        for original, detection in zip(raw, clean['detections']):
            if detection['label'] not in self.settings['labels']:
                raise ValueError('YOLO返回范围外类别')
            if detection['confidence'] < self.settings['thresholds'][detection['label']]:
                raise ValueError('YOLO返回低于配置阈值的检测')
            mask = cv2.imdecode(
                np.frombuffer(base64.b64decode(original['mask_png'], validate=True), np.uint8), 0
            )
            if mask is None or mask.shape != shape or not np.isin(mask, [0, 255]).all() or not mask.any():
                raise ValueError('YOLO实例掩膜无效')
            masks.append(mask > 0)
        overlay = cv2.imdecode(
            np.frombuffer(base64.b64decode(value['overlay_jpeg'], validate=True), np.uint8), cv2.IMREAD_COLOR
        )
        if overlay is None or overlay.shape[:2] != shape:
            raise ValueError('YOLO叠加图尺寸无效')
        clean['description'] = '本地实例分割：' + (
            '，'.join(d['label'] for d in clean['detections']) or '未发现支持类别'
        )
        return dict(
            **clean,
            masks=masks,
            overlay=overlay,
            model=value['model'],
            inference_ms=float(value['inference_ms']),
            source='yolo',
        )

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2.0)
        for stream in (self.process.stdin, self.process.stdout):
            if stream:
                stream.close()
        self.reader.join(timeout=1.0)
        self.log.close()
