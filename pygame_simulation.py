#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于检测结果的交通仿真：
1. 从 JSON 文件中读取各帧车辆 track 数据（检测结果）。
2. 模拟一条水平直路（单侧车道），采用两个车道，分别表示向右和向左行驶。
3. 利用全局字典保持车辆的连续运动（根据 track_id 更新车辆状态，不重置位置），
   并对连续未更新的车辆设定失联计数，超过一定帧数后才从仿真中移除，以缓解车辆突然消失的问题。
4. 根据车辆速度判断怠速：若检测速度低于阈值（现设为2.0）的车辆存在，
   则信号由绿灯切为黄灯（持续2秒）再变为红灯；若持续红灯超过5秒，则自动切回绿灯，
   避免车辆一直停在红灯状态。
5. 仿真界面中绘制道路、车道及右上角的信号灯和提示信息。
"""

import pygame
import json
import time
import sys
from pathlib import Path

# 窗口及背景设置
WIDTH, HEIGHT = 800, 600
BACKGROUND = (60, 160, 60)
ROAD_COLOR = (50, 50, 50)
LANE_COLOR = (255, 255, 255)
TEXT_COLOR = (0, 0, 0)

# 水平直路区域设置（整个窗口中心处的道路）
ROAD_RECT = pygame.Rect(0, HEIGHT//2 - 60, WIDTH, 120)
# 车道中心：采用两个车道
# lane0：车辆从左向右行驶，初始位置为道路左侧；
# lane1：车辆从右向左行驶，初始位置为道路右侧。
LANE_CENTERS = {
    0: (50, ROAD_RECT.centery - 20),
    1: (WIDTH - 50, ROAD_RECT.centery + 20)
}
# 车道运动方向
LANE_DIRECTIONS = {
    0: (1, 0),    # 向右
    1: (-1, 0)    # 向左
}

# 车辆显示参数
VEHICLE_WIDTH, VEHICLE_HEIGHT = 30, 15

# 信号灯参数
# 将怠速车辆阈值调低，现设为2.0（单位与检测数据一致，如 px/s）
IDLE_SPEED_THRESHOLD = 1.0
# 信号状态："GREEN", "YELLOW", "RED"
current_signal = "GREEN"
yellow_start_time = None  # 记录黄灯开始时间
red_start_time = None     # 记录红灯开始时间

# 每帧数据刷新间隔（秒）
FRAME_INTERVAL = 0.5

# 对于连续未更新的车辆，允许4帧（约2秒）内失联后再移除
MISSED_FRAME_THRESHOLD = 4

def update_signal(has_idle, current_time):
    """
    更新交通信号：
      - 若检测到怠速车辆且当前信号为 GREEN，则切换为 YELLOW 并记录黄灯开始时间；
      - 若处于 YELLOW 状态且持续2秒，则切换为 RED 并记录红灯开始时间；
      - 若处于 RED 状态且持续超过5秒，则自动切换回 GREEN；
      - 若当前无怠速车辆，则直接保持或切换为 GREEN。
    """
    global current_signal, yellow_start_time, red_start_time
    if has_idle:
        if current_signal == "GREEN":
            current_signal = "YELLOW"
            yellow_start_time = current_time
        elif current_signal == "YELLOW":
            if current_time - yellow_start_time >= 2.0:
                current_signal = "RED"
                red_start_time = current_time
        elif current_signal == "RED":
            if current_time - red_start_time >= 5.0:
                current_signal = "GREEN"
                yellow_start_time = None
                red_start_time = None
    else:
        current_signal = "GREEN"
        yellow_start_time = None
        red_start_time = None

SIGNAL_COLORS = {
    "GREEN": (50, 255, 50),
    "YELLOW": (255, 255, 50),
    "RED": (255, 50, 50)
}

# 使用两个车道模拟，车辆所属车道根据 track_id % 2 得出
class SimulationVehicle:
    def __init__(self, track_data):
        """
        初始化时根据 track_data（含 track_id、speed、signal 等）构造仿真车辆，
        车辆初始位置为对应车道中心。
        """
        self.track_id = int(track_data["track_id"])
        self.lane = self.track_id % 2
        init_pos = LANE_CENTERS[self.lane]
        self.x, self.y = init_pos  # 初始位置
        self.speed = track_data["speed"]
        self.signal = track_data["signal"]
        self.dir_vector = LANE_DIRECTIONS[self.lane]
        self.color = (255, 230, 50) if self.signal == "GREEN" else (180, 180, 180)
        self.missed_frames = 0    # 记录连续未更新的帧数
        self.last_seen = time.time()  # 最近更新时间

    def update_state(self, track_data, current_time):
        """
        根据最新检测数据更新车辆状态，不重置位置，
        同时重置失联计数和更新时间。
        """
        self.speed = track_data["speed"]
        self.signal = track_data["signal"]
        self.color = (255, 230, 50) if self.signal == "GREEN" else (180, 180, 180)
        self.last_seen = current_time
        self.missed_frames = 0

    def update(self, dt):
        # 红灯状态下车辆停止移动
        if current_signal == "RED":
            return
        dx = self.dir_vector[0] * self.speed * dt * 0.3
        dy = self.dir_vector[1] * self.speed * dt * 0.3
        self.x += dx
        self.y += dy
        # 当车辆移动出屏幕时循环回到另一侧
        if self.lane == 0 and self.x > WIDTH + VEHICLE_WIDTH:
            self.x = -VEHICLE_WIDTH
        elif self.lane == 1 and self.x < -VEHICLE_WIDTH:
            self.x = WIDTH + VEHICLE_WIDTH

    def draw(self, surface):
        rect = pygame.Rect(0, 0, VEHICLE_WIDTH, VEHICLE_HEIGHT)
        rect.center = (int(self.x), int(self.y))
        pygame.draw.rect(surface, self.color, rect)
        font = pygame.font.SysFont(None, 16)
        text = font.render(f"ID {self.track_id}", True, TEXT_COLOR)
        surface.blit(text, (self.x - VEHICLE_WIDTH // 2, self.y - VEHICLE_HEIGHT - 15))

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Traffic Simulation")
    clock = pygame.time.Clock()

    # 读取 JSON 文件中保存的所有帧 track 数据
    json_path = Path("tracks.json")
    try:
        with open(json_path, "r") as f:
            all_frames = json.load(f)
    except Exception as e:
        print(f"读取 JSON 数据失败: {e}")
        sys.exit(1)

    total_frames = len(all_frames)
    current_frame_idx = 0
    last_frame_update = time.time()

    # 用于保持车辆连续运动的字典，key: track_id，value: SimulationVehicle 对象
    sim_vehicles = {}

    running = True
    while running:
        current_time = time.time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 每隔 FRAME_INTERVAL 更新一次检测数据
        if current_time - last_frame_update >= FRAME_INTERVAL:
            current_frame_idx = (current_frame_idx + 1) % total_frames
            last_frame_update = current_time
            frame_data = all_frames[current_frame_idx]
            idle_exist = False
            updated_ids = set()
            # 遍历当前检测帧中的所有 track 数据
            for track in frame_data["tracks"]:
                track_id = int(track["track_id"])
                if track["speed"] < IDLE_SPEED_THRESHOLD:
                    idle_exist = True
                updated_ids.add(track_id)
                if track_id in sim_vehicles:
                    sim_vehicles[track_id].update_state(track, current_time)
                else:
                    sim_vehicles[track_id] = SimulationVehicle(track)
            # 对于未出现在当前帧中的车辆，增加失联计数
            remove_ids = []
            for vid, vehicle in sim_vehicles.items():
                if vid not in updated_ids:
                    vehicle.missed_frames += 1
                    # 若连续失联超过 MISSED_FRAME_THRESHOLD 帧，则计划移除该车辆
                    if vehicle.missed_frames > MISSED_FRAME_THRESHOLD:
                        remove_ids.append(vid)
            for vid in remove_ids:
                del sim_vehicles[vid]
            update_signal(idle_exist, current_time)

        # 绘制背景及道路
        screen.fill(BACKGROUND)
        pygame.draw.rect(screen, ROAD_COLOR, ROAD_RECT)
        # 绘制车道分界线（中间虚线）
        lane_line_y_top = ROAD_RECT.centery - 5
        lane_line_y_bot = ROAD_RECT.centery + 5
        for x in range(0, WIDTH, 40):
            pygame.draw.line(screen, LANE_COLOR, (x, lane_line_y_top), (x + 20, lane_line_y_top), 2)
            pygame.draw.line(screen, LANE_COLOR, (x, lane_line_y_bot), (x + 20, lane_line_y_bot), 2)

        dt = clock.get_time() / 1000.0
        for vehicle in sim_vehicles.values():
            vehicle.update(dt)
            vehicle.draw(screen)

        # 绘制信号灯（右上角）
        light_radius = 20
        light_pos = (WIDTH - 50, 50)
        pygame.draw.circle(screen, SIGNAL_COLORS[current_signal], light_pos, light_radius)
        font_large = pygame.font.SysFont(None, 30)
        signal_text = f"Signal: {current_signal}"
        text_surface = font_large.render(signal_text, True, TEXT_COLOR)
        screen.blit(text_surface, (WIDTH - 150, 80))
        if current_signal in ["YELLOW", "RED"]:
            prompt_surface = font_large.render("Detected idle vehicle(s)!", True, (255, 0, 0))
            screen.blit(prompt_surface, (WIDTH - 250, 120))

        # 显示当前帧编号
        font_small = pygame.font.SysFont(None, 24)
        frame_text = font_small.render(f"Frame {all_frames[current_frame_idx]['frame']}/{total_frames}", True, TEXT_COLOR)
        screen.blit(frame_text, (10, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
