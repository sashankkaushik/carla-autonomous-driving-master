#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down

    TAB          : change sensor position
    `            : next sensor
    [1-9]        : change to sensor [1-9]
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

# print(sys.path)
import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


try:
    sys.path.insert(0, '/home/saivinay/Documents/Carla_0.9.5/PythonAPI/carla')
except IndexError:
    pass
from agents.navigation.roaming_agent import RoamingAgent
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.ppo import PPO
from agents.navigation.ppo import Memory

from tensorboardX import SummaryWriter
from rewards.reward import Reward
import cv2




# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, actor_filter, actor_role_name='hero'):
        self.world = carla_world
        self.actor_role_name = actor_role_name
        self.map = self.world.get_map()
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = actor_filter
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

    def restart(self):
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        # print("Inside world.tick")
        self.hud.tick(self, clock)


    def render(self, display):
        # print("Inside world.render")
        self.camera_manager.render(display)
        self.hud.render(display)
        
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        # camera = self.world.spawn_actor(camera_bp, relative_transform, attach_to=my_vehicle)
        # print(camera_bp )

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()

# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            world.player.set_autopilot(self._autopilot_enabled)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock):
        # print("Inside keyboardControl.parse_events")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    currentIndex = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(currentIndex)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not (pygame.key.get_mods() & KMOD_CTRL):
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time())

            # self._control.throttle = 1.0            # added

            world.player.apply_control(self._control)  # if present works on manual mode

    def _parse_vehicle_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_walker_keys(self, keys, milliseconds):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = 5.556 if pygame.key.get_mods() & KMOD_SHIFT else 2.778
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame_number = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame_number = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame_number - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame_number, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self._camera_transforms = [
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            carla.Transform(carla.Location(x=1.6, z=1.7))]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '5000')
            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self.transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None \
            else self.sensors[index][0] != self.sensors[self.index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
            # print(self.sensor)
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            # print(array.shape)
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame_number)







########################################################################################################################################
############################################################# Changed code #############################################################
########################################################################################################################################


def parse_image(image, index, hud):
    '''
    To convert raw data from sensor output to numpy arrays
    '''    
    sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            # ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            # ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
    
    if index==3:
        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 3), 3))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(hud.dim) / 100.0
        lidar_data += (0.5 * hud.dim[0], 0.5 * hud.dim[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (hud.dim[0], hud.dim[1], 3)
        lidar_img = np.zeros(lidar_img_size)
        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
        surface = pygame.surfarray.make_surface(lidar_img)
        return lidar_img

    image.convert(sensors[index][1])
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    # print(array.shape)
    return array



#########################################################################################
class GetData():
    '''
    We store data from sensors and use for our agent
    '''
    def __init__(self):
        self.rgb = None
        self.depth = None
        self.seg = None
        self.lidar = None

        ################### Experimenting new cameras ###################
        self.front_left_cam = None
        self.front_right_cam = None 
        self.side_left_cam = None 
        self.side_right_cam = None 
        self.rear_cam = None 


    def store_data(self, sensor_data, sensor):
        if sensor=='rgb':
            self.rgb = sensor_data
            # print("rgb")        
        elif sensor=='depth':
            self.depth = sensor_data
            # print("depth")
        elif sensor=='seg':
            self.seg = sensor_data
            # print("seg")
        elif sensor=='lidar':
            self.lidar = sensor_data

        ################### Experimenting new cameras ###################
        elif sensor=='flc':
            self.front_left_cam = sensor_data
        elif sensor=='frc':
            self.front_right_cam = sensor_data
        elif sensor=='slc':
            self.side_left_cam = sensor_data
        elif sensor=='src':
            self.side_right_cam = sensor_data
        elif sensor=='rc':
            self.rear_cam = sensor_data

    
    def get_data(self):
        data = [self.rgb, self.depth, self.seg]

        ################### Experimenting new cameras ###################
        all_cams = [self.rgb, self.front_left_cam, self.front_right_cam, self.side_left_cam, self.side_right_cam, self.rear_cam]

        return data, all_cams
#########################################################################################


#########################################################################################

def perform_action(self, data):
        '''
        Performs action

        Input : Data = [image, segmentation, depth, ...]
        Return : control
        action = [throttle, steering_angle, steering_direction, brake, reverse_gear]
        '''
        action = self.agent.select_action(data, self.memory)
        print("Obtained action", action)

        #########################################
        # TODO
        # Add randomness in run_step of agent
        #########################################

        control = carla.VehicleControl()
        control.gear = 1

        # print(action[0].dtype)
        # a = float(action[0])
        # print(dtype(a))

        # Car Throttle 
        control.throttle = float(action[0])
        ################################################################
        # TODO
        # We need to convert angle into the units which carla uses 
        # steering direction will be a bool 0 for left and 1 for right
        # if action[2]:
        #     control.steer = function(action[1])
        # else:
        #     control.steer = -1*function(action[1])
        ################################################################
        # if action[2]:
        #     control.steer = action[1]
        # else:
        #     control.steer = -1*action[1]

        # brake is a bool
        # control.brake = action[3]
        # control.brake = 0

        # If needed there is an option for hand_brake as well
        # control.hand_brake

        # Car reverse gear
        # if action[4]:
        #     # self.carla_game._is_on_reverse = not self.carla_game._is_on_reverse   
        #     control.reverse = True
        # control.reverse = self.carla_game._is_on_reverse

        # control.reverse = 0

        return control, action

#########################################################################################

def distances(depth, rgb):
    '''
    TODO
    1. find the distance wrt car coordinate frame 
    2. check if the distances are in polar coordinate system and convert them to cartisian
    3.  
    '''

    dists = []

    # Get the coordinates of objects
    '''
    1. call yolo function on the image 
    coords = yolo(image) 
    '''
    # dummy
    coords = [[119,241],[258,353]]

    # Convert depth map values to real distances
    '''
    1. See the math required and the variables needed to convert disparty map to distances
    
    get intrinsic matrix and which is made of focal length and center coordinates of image CarlaSettings.ini
    
        Focus_length = ImageSizeX /(2 * tan(CameraFOV * π / 360))
        Center_X = ImageSizeX / 2
        Center_Y = ImageSizeY / 2
    '''
    



    # mask the depth map get the minmum distances 
    '''
    1. using the coordinates get the mask parts and find minimum from each patch
    '''
    # roi = np.mean(roi, axis=2)
    # depth1 = np.mean(depth, axis=2)
    # depth1 = np.expand_dims(depth1, axis=-1)
    depth1 = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
    roi = depth[coords[0][0]:coords[1][0], coords[0][1]:coords[1][1]]

    # depth[coords[0][0],coords[0][1]]
    cv2.rectangle(depth1, (coords[0][1], coords[0][0]), (coords[1][1], coords[1][0]), (0,255,0), 15)
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(depth,'OpenCV Tuts!',(10,500), font, 6, (200,255,155), 13, cv2.LINE_AA)
    # print(depth.shape)
    
    temp = np.hstack(roi)
    print(depth1.shape)
    print(np.amin(temp))
    cv2.imshow("roi", roi)
    # cv2.imshow("rgb", rgb)
    cv2.imshow("mean_depth", depth1)
    # cv2.imshow("depth",depth)
    cv2.waitKey(500)


    return dists 
     
#########################################################################################


def get_coordinates(image):



    return x,y,h,w

#########################################################################################

class Vehicle():

    
    self.location = None 
    self.velocity = None
    self.acceleration = None
    self.lane = None
    self.ego = None

    
    def __init__(self, args):
        '''
        From sensors get the velocity of ego vehicle
        From the environmnet class we get the velocities of other objects
        '''
        ego = args.ego

    '''
    Predict path
    '''

#########################################################################################


class Environment(object):
    '''
    We pass the vehicle objects to fill the environment
    '''

    def __init__(self):
        '''
        Initialize an empty environment
        Number of lanes = Assign based on lane detection 
        The size of environment = [(Till the maximum predictable depth / Precision), (Total width of consideration / Precision)] 
        '''

        # x=rows (vertical pixels) ; y=columns (horizontal pixels) 
        self.precision = None
        self.grid = np.ones([None,None])
        self.lanes = None
        self.vehicle_dims = [3*100/precision,2*100/precision]
    
    def get_coordinates_and_distances(self, camera_location, camera=None):
        '''
        We get the distance of each object and usign FOV of the particular camera, we calculate the coordinates of it on the 
        grid environment
            - We detect objects and distances
            - Get the locations
            - Return values based on single coordinate system i.e, based on camera and location give the coordinates absolute values
              wrt to the center of mass of the car

        '''
    
    
    def sensor_fusion(self, number_of_cameras, dist):
        '''
        Arguments :  Map of camera and the blacked out coordinates in the camera 
        
        dist = []
        For camera 0
        dist[0] = [ [[startcoordinates], [endcoordinates]],  [[], []] , ]  
        
        what about traffic signs
        '''

        # Place objects
        for i in range(number_of_cameras):

            for j in range(dist[i]):
                [[x1, y1], [x2, y2]] = dist[i][j]

                grid[x1:x2, y1:y2] = -1
        
        # Place lanes 
    
    def map_coordinates(self, number_of_cameras):

        dist = []
        for i in range(number_of_cameras):
            dist.append(self.get_coordinates_and_distances(i))



    def upate_environment(self):
        '''
        We call this each instant to change the environment conditions like the lanes, the grid size, the occupancy 
        '''

#########################################################################################




# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================

def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        ##### Default #####
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args.filter, args.rolename)
        controller = KeyboardControl(world, args.autopilot)

        clock = pygame.time.Clock()


        world_=client.get_world()
        blueprint_library = world_.get_blueprint_library()
        # bp=random.choice(blueprint_library.filter('vehicle'))
        bp=blueprint_library.filter('vehicle.model3')   # change this to fix a car

        transform = random.choice(world_.get_map().get_spawn_points())
        vehicle = world.player
        ##### Default #####


        ################################
        # to get the sensor inputs
        p = GetData()
        ################################


        ######################################################################
        
        # Stores all the sensor inputs whenever sensor captures input in data 
        camera_bp_rgb = blueprint_library.find('sensor.camera.rgb')
        camera_transform_rgb= carla.Transform(carla.Location(x=1.7, z=1.7))
        camera_rgb = world_.spawn_actor(camera_bp_rgb, camera_transform_rgb, attach_to=vehicle)
        # cc.LogarithmicDepth(camera_rgb).listen(lambda image: image.save_to_disk('./output_rgb/%06d.png' % image.frame_number))
        camera_rgb.listen(lambda image: p.store_data(image, 'rgb'))      
        # print(camera_transform_rgb.focal_distance())  
        print('created %s' % camera_rgb.type_id)

        camera_bp_depth = blueprint_library.find('sensor.camera.depth')
        camera_transform_depth= carla.Transform(carla.Location(x=1.5, z=2.4))
        camera_depth = world_.spawn_actor(camera_bp_depth, camera_transform_depth, attach_to=vehicle)
        # camera_depth.listen(lambda image: image.save_to_disk('./output_depth/%06d.png' % image.frame_number))
        camera_depth.listen(lambda image: p.store_data(image, 'depth'))        
        print('created %s' % camera_depth.type_id)

        camera_bp_ss = blueprint_library.find('sensor.camera.semantic_segmentation')
        camera_transform_ss= carla.Transform(carla.Location(x=1.9, z=2.4))
        camera_ss = world_.spawn_actor(camera_bp_ss, camera_transform_ss, attach_to=vehicle)
        # camera_ss.listen(lambda image: cc.CityScapesPalette(image).save_to_disk('./output_ss/%06d.png' % image.frame_number))
        camera_ss.listen(lambda image: p.store_data(image, 'seg'))
        print('created %s' % camera_ss.type_id)

        # Lidar
        # camera_bp_lid = blueprint_library.find('sensor.lidar.ray_cast')
        # camera_transform_lid= carla.Transform(carla.Location(x=1.9, z=2.4))
        # camera_lid = world_.spawn_actor(camera_bp_lid, camera_transform_lid, attach_to=vehicle)
        # # camera_ss.listen(lambda image: cc.CityScapesPalette(image).save_to_disk('./output_ss/%06d.png' % image.frame_number))
        # camera_lid.listen(lambda image: p.store_data(image, 'lidar'))
        # print('created %s' % camera_lid.type_id)

        ######################################################################
        ################### Experimenting new cameras ###################
        '''
        camera_bp_flc = blueprint_library.find('sensor.camera.rgb')
        camera_transform_flc= carla.Transform(carla.Location(x=1.7, z=1.2, y=-1))
        camera_flc = world_.spawn_actor(camera_bp_flc, camera_transform_flc, attach_to=vehicle)
        # cc.LogarithmicDepth(camera_rgb).listen(lambda image: image.save_to_disk('./output_rgb/%06d.png' % image.frame_number))
        camera_flc.listen(lambda image: p.store_data(image, 'flc')) 

        camera_bp_frc = blueprint_library.find('sensor.camera.rgb')
        camera_transform_frc= carla.Transform(carla.Location(x=1.7, z=1.2, y=1))
        camera_frc = world_.spawn_actor(camera_bp_frc, camera_transform_frc, attach_to=vehicle)
        # cc.LogarithmicDepth(camera_rgb).listen(lambda image: image.save_to_disk('./output_rgb/%06d.png' % image.frame_number))
        camera_frc.listen(lambda image: p.store_data(image, 'frc')) 

        camera_bp_slc = blueprint_library.find('sensor.camera.rgb')
        camera_transform_slc= carla.Transform(carla.Location(x=0, z=1.2), carla.Rotation(yaw=-90))
        camera_slc = world_.spawn_actor(camera_bp_slc, camera_transform_slc, attach_to=vehicle)
        # cc.LogarithmicDepth(camera_rgb).listen(lambda image: image.save_to_disk('./output_rgb/%06d.png' % image.frame_number))
        camera_slc.listen(lambda image: p.store_data(image, 'slc')) 

        camera_bp_src = blueprint_library.find('sensor.camera.rgb')
        camera_transform_src= carla.Transform(carla.Location(x=0, z=1.2), carla.Rotation(yaw=90))
        camera_src = world_.spawn_actor(camera_bp_src, camera_transform_src, attach_to=vehicle)
        # cc.LogarithmicDepth(camera_rgb).listen(lambda image: image.save_to_disk('./output_rgb/%06d.png' % image.frame_number))
        camera_src.listen(lambda image: p.store_data(image, 'src')) 

        camera_bp_rc = blueprint_library.find('sensor.camera.rgb')
        camera_transform_rc= carla.Transform(carla.Location(x=-1.7, z=1.2), carla.Rotation(yaw=180))
        camera_rc = world_.spawn_actor(camera_bp_rc, camera_transform_rc, attach_to=vehicle)
        # cc.LogarithmicDepth(camera_rgb).listen(lambda image: image.save_to_disk('./output_rgb/%06d.png' % image.frame_number))
        camera_rc.listen(lambda image: p.store_data(image, 'rc'))         
        '''
        
        
        # initializing train
        # train = Train(args) # change this 

        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock):
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

            # we get the current stored data
            data, all_cams = p.get_data()

            if not any(elem is None for elem in data):
                 
                '''previous code'''
                rgb = parse_image(data[0], 0, hud)
                depth = parse_image(data[1], 1, hud)
                seg = parse_image(data[2], 2, hud)
                # lidar = parse_image(data[3], 3, hud)
                # lidar = data[3]

                # print(rgb.shape, depth.shape, seg.shape)
                np.set_printoptions(threshold=np.inf)
                # cv2.imshow("rgb", rgb)
                # cv2.imshow("depth",depth)
                # cv2.waitKey(500)
                

                # Merge depth with rgb and get the distances
                output_distances = distances(depth,rgb)

                '''
                ################### Experimenting new cameras ###################
                rgb = parse_image(all_cams[0],0)
                flc = parse_image(all_cams[1],0)
                frc = parse_image(all_cams[2],0)
                slc = parse_image(all_cams[3],0)
                src = parse_image(all_cams[4],0)
                rc = parse_image(all_cams[5],0)

                # cv2.imshow("1.7, z=1.2, y=0", rgb)
                # cv2.waitKey(10)
                # cv2.imshow("x=1.7, z=1.2, y=-1", flc)
                # cv2.waitKey(10)
                # cv2.imshow("x=1.7, z=1.2, y=1", frc)
                # cv2.waitKey(10)
                # cv2.imshow("0, 1.2, -90", slc)
                # cv2.waitKey(10)
                # cv2.imshow("0, 1.2, 90", src)
                # cv2.waitKey(10)
                # cv2.imshow("-1.7, 1.2, 180", rc)
                # cv2.waitKey(10)


                # print(vehicle.get_velocity(), vehicle.get_acceleration(), vehicle.get_location())
                # print(vehicle.forward_speed())
                
                '''
                

    finally:

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument("--agent", type=str,
                           choices=["Roaming", "Basic","PPO"],
                           help="select which agent to run",
                           default="PPO")

    argparser.add_argument( '--save_freq', type = int, default=5, help='time steps after which network weights are saved')
    argparser.add_argument( '--checkpoints', type=str, default='./checkpoints', help='path to checkpoints folder')
    argparser.add_argument( '--summary_path', type=str, default='./summary', help='path to summary folder')
    argparser.add_argument( '--update', type=int, default=10, help='the update frequency of the agent')



    
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
    '''
    import argparse

    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--imageA', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--imageB', type=str,
                        help='path to a test image or folder of images', required=True)                    
    
    args = parser.parse_args()
    # val = mse(args.imageA,args.imageB)
    prediction = cv2.imread(args.imageA,0)
    actual = cv2.imread(args.imageB,0)
    # actual = parse_image(actual,1,0)
    pred_stack, actual_stack = np.hstack(prediction), np.hstack(actual)

    meanp,stdp = np.mean(pred_stack), np.std(pred_stack)
    meana,stda = np.mean(actual_stack), np.std(actual_stack)

    nor_prediction = np.asarray([(np.array(xi)-meanp)/stdp for xi in prediction])
    nor_actual = np.asarray([(np.array(xi)-meana)/stda for xi in actual])

    # err = np.sqrt(np.sum((nor_prediction.astype("float") - nor_actual.astype("float")) ** 2))
    err = np.sqrt(((nor_prediction - nor_actual)**2).mean(axis=None))
    err /= float(nor_prediction.shape[0] * nor_prediction.shape[1])
    print("RMSE = ", err)
    
    cv2.imshow('prediction',nor_prediction)
    cv2.imshow('actual',nor_actual)

    cv2.waitKey(10000)
    cv2.destroyAllWindows()
    '''
