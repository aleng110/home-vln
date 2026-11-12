"""仿真 RGB-D 图像解码。"""

from __future__ import annotations

import cv2
import numpy as np
from sensor_msgs.msg import Image


def decode_ros_image(msg: Image) -> np.ndarray:
    """按 ROS 图像编码和行步长还原 RGB 或深度数组。"""
    encoding = msg.encoding.lower()
    color_encodings = {
        "bgr8": (np.uint8, 3),
        "rgb8": (np.uint8, 3),
        "bgra8": (np.uint8, 4),
        "rgba8": (np.uint8, 4),
        "mono8": (np.uint8, 1),
    }
    depth_encodings = {"32fc1": np.float32, "16uc1": np.uint16}

    if encoding in color_encodings:
        dtype, channels = color_encodings[encoding]
    elif encoding in depth_encodings:
        dtype, channels = depth_encodings[encoding], 1
    else:
        raise ValueError(f"unsupported image encoding: {msg.encoding}")

    dtype = np.dtype(dtype).newbyteorder(">" if msg.is_bigendian else "<")
    row_items = int(msg.step) // dtype.itemsize
    required_items = row_items * int(msg.height)
    flat = np.frombuffer(msg.data, dtype=dtype, count=required_items)
    rows = flat.reshape(int(msg.height), row_items)
    cropped = rows[:, : int(msg.width) * channels]
    if channels > 1:
        image = cropped.reshape(int(msg.height), int(msg.width), channels)
    else:
        image = cropped

    if encoding == "rgb8":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif encoding == "rgba8":
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    elif encoding == "bgra8":
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif encoding == "mono8":
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif encoding == "16uc1":
        image = image.astype(np.float32) / 1000.0

    return np.asarray(image)
