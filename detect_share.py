# detect_share.py
# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
此模块用于在实际应用中由detect.py更新最新的tracks数据和路口信号状态，
并提供线程安全的接口供其他模块（例如pygame仿真模块）调用。
注：在实际接入YOLOv5+DeepSORT时，detect.py检测得到的track对象要求至少包含:
      - track_id 属性（标识ID）
      - to_ltrb() 方法，返回当前目标的边界框 [x1, y1, x2, y2]
      - 可选：你还可以将speed和signal属性写入track对象，供后续使用
"""

import threading

# 使用全局变量存储共享数据
_current_tracks = []  # 保存最新的track列表
_signal = "UNKNOWN"  # 保存当前路口信号状态，例如 "GREEN" 或 "RED"

# 创建一个线程锁，以确保数据更新和读取时的原子性
_lock = threading.Lock()


def update_current_tracks(new_tracks):
    """
    更新全局tracks数据
    :param new_tracks: 当前帧检测跟踪到的track对象列表，这些对象由YOLOv5+DeepSORT生成，
                       每个对象应至少包含 track_id 和 to_ltrb() 方法。
    """
    global _current_tracks
    with _lock:
        _current_tracks = new_tracks


def get_current_tracks():
    """
    获取最新的tracks数据
    :return: 当前共享的track对象列表
    """
    with _lock:
        return _current_tracks  # 可根据需要返回拷贝，比如 list(_current_tracks)


def update_traffic_signal(signal):
    """
    更新全局路口信号状态
    :param signal: 字符串，如 "GREEN" 或 "RED"
    """
    global _signal
    with _lock:
        _signal = signal


def get_traffic_signal():
    """
    获取当前的路口信号状态
    :return: 当前信号状态（例如 "GREEN" 或 "RED"）
    """
    with _lock:
        return _signal


if __name__ == "__main__":
    # 简单测试：模拟两个track对象（实际应用中由DeepSORT产生）
    class DummyTrack:
        def __init__(self, track_id, bbox):
            self.track_id = track_id
            self._bbox = bbox

        def to_ltrb(self):
            return self._bbox


    dummy_tracks = [DummyTrack(1, [400, 300, 440, 340]), DummyTrack(2, [100, 100, 140, 140])]
    update_current_tracks(dummy_tracks)
    update_traffic_signal("GREEN")

    print("Current Tracks:")
    for t in get_current_tracks():
        print("Track ID:", t.track_id, "BBox:", t.to_ltrb())

    print("Traffic Signal State:", get_traffic_signal())
