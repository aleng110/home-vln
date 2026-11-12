"""常驻 Conda 分割进程：接收 PNG，返回检测框、原尺寸掩膜和标注图。

标准输出仅供 JSONL 协议使用，模型日志写入标准错误。"""

import argparse
import base64
from contextlib import redirect_stdout
import ctypes
import json
import os
from pathlib import Path
import signal
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--parent-pid', type=int, required=True)
    args = parser.parse_args()
    # 父进程退出后同步结束工作进程，释放模型资源。
    if sys.platform == 'linux':
        ctypes.CDLL(None).prctl(1, signal.SIGTERM)
        if os.getppid() != args.parent_pid:
            return
    protocol = sys.stdout

    def send(value):
        protocol.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        protocol.flush()

    with redirect_stdout(sys.stderr):
        import cv2
        import numpy as np
        import torch
        from ultralytics import YOLOE
        from yolo_settings import load_settings

        root = Path(args.project_root)
        settings = load_settings(root / 'config/yolo_scene_classes.yaml')
        if settings['device'] != 'cpu' and not torch.cuda.is_available():
            raise RuntimeError('YOLO工作进程未检测到CUDA')
        torch.set_num_threads(4)
        cv2.setNumThreads(1)
        model = YOLOE(str(root / 'work/yolo/assets' / settings['model']))
        model.to('cpu' if settings['device'] == 'cpu' else 'cuda:' + settings['device'])
        model.set_classes(settings['prompts'])
        options = dict(
            device=settings['device'],
            imgsz=settings['image_size'],
            conf=min(settings['thresholds'].values()),
            retina_masks=True,
            max_det=20,
            verbose=False,
        )
        model.predict(np.zeros((424, 512, 3), np.uint8), **options)
    send(
        dict(
            type='ready',
            model=settings['model'],
            labels=settings['labels'],
            device=settings['device'],
            thresholds=settings['thresholds'],
        )
    )

    def encode(image, extension):
        ok, encoded = cv2.imencode(extension, image)
        if not ok:
            raise ValueError('图像编码失败')
        return base64.b64encode(encoded).decode('ascii')

    for line in sys.stdin:
        request_id = None
        try:
            if len(line) > 12_000_000:
                raise ValueError('请求图像过大')
            request = json.loads(line)
            request_id = request['id']
            encoded = base64.b64decode(request['image_png'], validate=True)
            image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[0] * image.shape[1] > 2_000_000:
                raise ValueError('无效或过大的RGB图像')
            started = time.perf_counter()
            with redirect_stdout(sys.stderr):
                result = model.predict(image, **options)[0]
                detections = []
                if result.masks is None and len(result.boxes):
                    raise ValueError('分割模型返回了框但缺少实例掩膜')
                for i, box in enumerate(result.boxes):
                    index = int(box.cls.item())
                    label = settings['labels'][index]
                    confidence = float(box.conf.item())
                    if confidence < settings['thresholds'][label]:
                        continue
                    mask = (result.masks.data[i].cpu().numpy() > 0.5).astype(np.uint8) * 255
                    if mask.shape != image.shape[:2]:
                        raise ValueError('分割掩膜不是原图分辨率')
                    height, width = mask.shape
                    bounds = np.clip(
                        box.xyxy[0].cpu().numpy() / [width, height, width, height], 0, 1
                    ).tolist()
                    detections.append(
                        dict(
                            label=label,
                            raw_label=settings['prompts'][index],
                            confidence=confidence,
                            bbox=bounds,
                            mask_png=encode(mask, '.png'),
                        )
                    )
                # 只绘制实际交给任务端的检测结果。
                overlay = image.copy()
                for detection in detections:
                    mask = (
                        cv2.imdecode(np.frombuffer(base64.b64decode(detection['mask_png']), np.uint8), 0) > 0
                    )
                    overlay[mask] = (overlay[mask] * 0.6 + np.array([40, 210, 70]) * 0.4).astype(np.uint8)
                    scale = [image.shape[1], image.shape[0], image.shape[1], image.shape[0]]
                    x0, y0, x1, y1 = (np.array(detection['bbox']) * scale).astype(int)
                    cv2.rectangle(overlay, (x0, y0), (x1, y1), (40, 210, 70), 2)
                    cv2.putText(
                        overlay,
                        detection['label'] + ' %.2f' % detection['confidence'],
                        (x0, max(16, y0 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (40, 210, 70),
                        1,
                    )
            send(
                dict(
                    type='result',
                    id=request_id,
                    model=settings['model'],
                    width=image.shape[1],
                    height=image.shape[0],
                    detections=detections,
                    inference_ms=(time.perf_counter() - started) * 1000,
                    overlay_jpeg=encode(overlay, '.jpg'),
                )
            )
        except Exception as exc:
            send(dict(type='error', id=request_id, message=str(exc)))


if __name__ == '__main__':
    main()
