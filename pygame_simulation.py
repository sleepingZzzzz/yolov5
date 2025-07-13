#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于检测结果的交通仿真：
1. 从JSON文件中读取检测数据。
2. 模拟一条水平直路，设置两个车道（左向右和右向左）。
3. 利用连续的track数据实现车辆平滑运动。
4. 信号灯状态机带有倒计时：
   - 当状态转入GREEN时，根据当时怠速车辆数设置动态绿灯时长：
         dynamic_green_duration = fixed_green_duration + (idle_count × add_per_car)
     该时长在本周期内保持不变；
   - GREEN倒计时到0后转为YELLOW，YELLOW持续固定时间后转RED，
     RED持续固定时间后再转GREEN，并重新计算动态绿灯时长。
5. 界面显示当前信号状态以及倒计时（秒）。
"""

import pygame
import json
import time
import sys
from pathlib import Path

# -------------------- 窗口与道路设置 --------------------
WIDTH, HEIGHT = 800, 600
BACKGROUND = (60, 160, 60)
ROAD_COLOR = (50, 50, 50)
LANE_COLOR = (255, 255, 255)
TEXT_COLOR = (0, 0, 0)
ROAD_RECT = pygame.Rect(0, HEIGHT // 2 - 60, WIDTH, 120)

# -------------------- 车道参数 --------------------
# 根据track_id % 2分配车道，设置车道中心与行驶方向
LANE_CENTERS = {
    0: (50, ROAD_RECT.centery - 20),  # lane0：车辆从左向右
    1: (WIDTH - 50, ROAD_RECT.centery + 20)  # lane1：车辆从右向左
}
LANE_DIRECTIONS = {
    0: (1, 0),
    1: (-1, 0)
}
VEHICLE_WIDTH, VEHICLE_HEIGHT = 30, 15

# -------------------- 信号灯状态机参数 --------------------
fixed_green_duration = 5.0  # 基础绿灯时长（秒）
add_per_car = 3.0  # 每辆怠速车增加的时间（秒）
fixed_yellow_duration = 2.0  # 黄灯持续时间（秒）
fixed_red_duration = 5.0  # 红灯持续时间（秒）

# 当前信号状态，初始为GREEN；状态可能为 "GREEN"、"YELLOW"、"RED"
current_state = "GREEN"
state_start_time = None  # 记录当前状态开始的时间（秒）
state_duration = fixed_green_duration  # 本次状态周期的预设持续时长

# -------------------- 其他参数 --------------------
IDLE_SPEED_THRESHOLD = 2.0  # 速度低于此数视为怠速（单位：px/s）
FRAME_INTERVAL = 0.5  # 每0.5秒更新一次检测数据
MISSED_FRAME_THRESHOLD = 4  # 连续丢失4帧后移除车辆

# 定义信号灯颜色字典
SIGNAL_COLORS = {
    "GREEN": (50, 255, 50),
    "YELLOW": (255, 255, 50),
    "RED": (255, 50, 50)
}


# -------------------- 仿真车辆类 --------------------
class SimulationVehicle:
    def __init__(self, track_data):
        """
        初始化时根据track_data为车辆分配车道、初始位置、速度和信号。
        """
        self.track_id = int(track_data["track_id"])
        self.lane = self.track_id % 2
        self.x, self.y = LANE_CENTERS[self.lane]
        self.speed = track_data["speed"]
        self.signal = track_data["signal"]
        self.dir_vector = LANE_DIRECTIONS[self.lane]
        # 根据检测信号设定颜色（本例初始仅用绿色和灰色表示）
        self.color = (255, 230, 50) if self.signal == "GREEN" else (180, 180, 180)
        self.missed_frames = 0
        self.last_seen = time.time()

    def update_state(self, track_data, current_time):
        self.speed = track_data["speed"]
        self.signal = track_data["signal"]
        self.color = (255, 230, 50) if self.signal == "GREEN" else (180, 180, 180)
        self.last_seen = current_time
        self.missed_frames = 0

    def update(self, dt):
        # 红灯状态下车辆不移动
        if current_state == "RED":
            return
        dx = self.dir_vector[0] * self.speed * dt * 0.3
        dy = self.dir_vector[1] * self.speed * dt * 0.3
        self.x += dx
        self.y += dy
        # 当车辆移出屏幕，从另一侧进入
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


# -------------------- 信号灯状态机函数 --------------------
def update_signal(current_time, idle_count):
    """
    状态转换逻辑：
    1. 当进入GREEN状态时（即刚转为GREEN），根据当前检测到的怠速车辆数计算动态绿灯时长：
         state_duration = fixed_green_duration + idle_count * add_per_car
       此后整个GREEN周期内该时长保持不变，不随后续车辆变化而动态变化。
    2. GREEN状态到期后转为YELLOW，持续固定时间；YELLOW到期后转RED，持续固定时间；
       RED到期后转为GREEN，并重新计算动态绿灯时长。
    """
    global current_state, state_start_time, state_duration

    elapsed = current_time - state_start_time
    if current_state == "GREEN":
        if elapsed >= state_duration:
            current_state = "YELLOW"
            state_start_time = current_time
            state_duration = fixed_yellow_duration
    elif current_state == "YELLOW":
        if elapsed >= state_duration:
            current_state = "RED"
            state_start_time = current_time
            state_duration = fixed_red_duration
    elif current_state == "RED":
        if elapsed >= state_duration:
            # 当从RED转入GREEN时，重新确定动态绿灯时长，根据当前怠速车辆数
            current_state = "GREEN"
            state_start_time = current_time
            state_duration = fixed_green_duration + idle_count * add_per_car


# -------------------- 主函数 --------------------
def main():
    global state_start_time, state_duration, current_state
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Traffic Simulation")
    clock = pygame.time.Clock()

    # 读取检测数据（JSON文件路径请根据实际情况设置）
    json_path = Path("result/traffic_analysis/tracks.json")
    try:
        with open(json_path, "r") as f:
            all_frames = json.load(f)
    except Exception as e:
        print(f"读取JSON数据失败: {e}")
        sys.exit(1)

    total_frames = len(all_frames)
    current_frame_idx = 0
    last_frame_update = time.time()

    sim_vehicles = {}
    # 初始化信号灯状态：GREEN，并记录状态开始时间
    current_state = "GREEN"
    state_start_time = time.time()
    # 初始计算动态绿灯时长（在第一帧时统计怠速车数量）
    initial_idle_count = 0
    state_duration = fixed_green_duration + initial_idle_count * add_per_car

    running = True
    while running:
        current_time = time.time()
        # 处理退出事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 每隔FRAME_INTERVAL更新检测数据
        if current_time - last_frame_update >= FRAME_INTERVAL:
            current_frame_idx = (current_frame_idx + 1) % total_frames
            last_frame_update = current_time
            frame_data = all_frames[current_frame_idx]
            updated_ids = set()
            for track in frame_data["tracks"]:
                track_id = int(track["track_id"])
                updated_ids.add(track_id)
                if track_id in sim_vehicles:
                    sim_vehicles[track_id].update_state(track, current_time)
                else:
                    sim_vehicles[track_id] = SimulationVehicle(track)
            # 对于未在当前帧中出现的车辆增加失联计数，并删除连续丢失车辆
            remove_ids = []
            for vid, vehicle in sim_vehicles.items():
                if vid not in updated_ids:
                    vehicle.missed_frames += 1
                    if vehicle.missed_frames > MISSED_FRAME_THRESHOLD:
                        remove_ids.append(vid)
            for vid in remove_ids:
                del sim_vehicles[vid]

        # 统计当前怠速车辆数：判断条件使用速度低于IDLE_SPEED_THRESHOLD
        idle_count = sum(1 for v in sim_vehicles.values() if v.speed < IDLE_SPEED_THRESHOLD)

        # 调用信号灯状态机更新（注意：状态转换时动态绿灯时长只在GREEN初始时计算）
        update_signal(current_time, idle_count)

        # 绘制背景和道路
        screen.fill(BACKGROUND)
        pygame.draw.rect(screen, ROAD_COLOR, ROAD_RECT)
        lane_line_y_top = ROAD_RECT.centery - 5
        lane_line_y_bot = ROAD_RECT.centery + 5
        for x in range(0, WIDTH, 40):
            pygame.draw.line(screen, LANE_COLOR, (x, lane_line_y_top), (x + 20, lane_line_y_top), 2)
            pygame.draw.line(screen, LANE_COLOR, (x, lane_line_y_bot), (x + 20, lane_line_y_bot), 2)

        dt = clock.get_time() / 1000.0
        for veh in sim_vehicles.values():
            veh.update(dt)
            veh.draw(screen)

        # 绘制信号灯（右上角）
        light_radius = 20
        light_pos = (WIDTH - 50, 50)
        pygame.draw.circle(screen, SIGNAL_COLORS[current_state], light_pos, light_radius)
        font_large = pygame.font.SysFont(None, 30)
        state_text = f"Signal: {current_state}"
        state_surface = font_large.render(state_text, True, TEXT_COLOR)
        screen.blit(state_surface, (WIDTH - 150, 80))

        # 根据当前状态显示倒计时
        elapsed = current_time - state_start_time
        remaining = max(state_duration - elapsed, 0)
        countdown_text = f"Countdown: {remaining:.1f}s"
        countdown_surface = font_large.render(countdown_text, True, (255, 255, 255))
        screen.blit(countdown_surface, (WIDTH - 150, 120))

        # 在GREEN状态时显示当前绿灯时长设定
        if current_state == "GREEN":
            green_info = f"Green Duration: {state_duration:.1f}s (Idle: {idle_count})"
            green_surface = font_large.render(green_info, True, (0, 255, 255))
            screen.blit(green_surface, (WIDTH - 320, 160))

        # 显示帧号
        font_small = pygame.font.SysFont(None, 24)
        frame_info = f"Frame {all_frames[current_frame_idx]['frame']}/{total_frames}"
        frame_surface = font_small.render(frame_info, True, TEXT_COLOR)
        screen.blit(frame_surface, (10, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
