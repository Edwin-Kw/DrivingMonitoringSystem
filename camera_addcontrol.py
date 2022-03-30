#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

#"person":1
#"bicycle":2
#"car":3
#"motorcycle":4
#"bus":6
#"train":7
#"truck":8
#"traffic light":10
#"stop sign":13

import glob
import os
import sys
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib

from numpy import quantile
import tensorflow as tf
import math
import weakref
import requests
#tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
""" gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True) """
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random
import json
import time
try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    
    from pygame.locals import K_LEFT
    
    from pygame.locals import K_RIGHT
    
    from pygame.locals import K_SPACE
    
    from pygame.locals import K_UP
    from pygame.locals import K_a
    
    from pygame.locals import K_d
    
    from pygame.locals import K_q
    
    from pygame.locals import K_s
    
    from pygame.locals import K_w
    
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import ops as utils_ops

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile
MODEL_PATH = 'C:\\Users\\Acer\\.keras\\datasets\\ssd_mobilenet_v2_coco_2018_03_29'
LABEL_PATH = 'C:\\Users\\Acer\\.keras\\datasets\\mscoco_label_map.pbtxt'
CAMERA_DEPTH = 0.5
#camera_constant = 0.0

important_list = [1,2,3,4,6,7,8]
classes =  {1:"person",
            2:"bicycle",
            3:"car",
            4:"motorcycle",
            6:"bus",
            7:"train",
            8:"truck"}
DRIVER_ID = "001"
#"traffic light":10
#"stop sign":13

#DRIVER_ID, DISPALY, CROSSING, speed, location, distance_set, distance_classes
class ImageRecord:
    def __init__(self,DRIVER_ID, DISPALY, speed, CROSSING, location, distance_set, distance_classes, warning, warningMessage, timestring, timenow):
        self.DRIVER_ID = DRIVER_ID
        self.DISPALY = DISPALY
        self.speed = speed
        self.CROSSING = CROSSING
        self.location = location
        self.distance_set = distance_set
        self.distance_classes = distance_classes
        self.warning = warning
        self.warningMessage = warningMessage
        self.timestring = timestring
        self.timenow = timenow

    def statusReturn(self):
        data = {"DRIVER_ID" : self.DRIVER_ID, 
                "speed": self.speed, 
                "CROSSING": self.CROSSING, 
                "location": self.location,
                "distance_set": self.distance_set,
                "distance_classes": self.distance_classes,
                "warning": self.warning,
                "warningMessage": self.warningMessage,
                "timestring": self.timestring,
                "timenow": self.timenow
                }

        jsonData = json.dumps(data)
        return data

    def imageReturn(self):
        return self.DISPALY
    
    def driverReturn(self):
        return self.DRIVER_ID

class World(object):
    def __init__(self, carla_world,vehicle):
        self.world = carla_world
        self.sync = True
        self.player = vehicle
        #self._actor_filter = args.filter
        self.modify_vehicle_physics(self.player)

    """ def restart(self):
        #blueprint = random.choice(get_actor_blueprints(self.world, self._actor_filter, self._actor_generation))
        
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = spawn_points[0]
            #spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player) """
        
    def modify_vehicle_physics(self, actor):
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

class KeyboardControl(object):
    """Class that handles keyboard input."""
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_autopilot(self._autopilot_enabled)
            world.player.set_light_state(self._lights)
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        

    def parse_events(self, client, world, clock, sync_mode):
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                
        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
                # Set automatic control-related vehicle lights
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else: # Remove the Brake flag
                    current_lights &= ~carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else: # Remove the Reverse flag
                    current_lights &= ~carla.VehicleLightState.Reverse
                if current_lights != self._lights: # Change the light state only if necessary
                    self._lights = current_lights
                    world.player.set_light_state(carla.VehicleLightState(self._lights))
            world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            self._control.throttle = min(self._control.throttle + 0.3, 1.00)
        else:
            self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            self._control.brake = min(self._control.brake + 0.2, 1)
        else:
            self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.hand_brake = keys[K_SPACE]

    

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

def get_font0():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


class LaneInvasionSensor(object):
    def __init__(self, parent_actor,display):
        self.sensor = None
        self.display = display
        self.doubleline = False
        self.text = ""
        # If the spawn object is not a vehicle, we cannot use the Lane Invasion Sensor
        if parent_actor.type_id.startswith("vehicle."):
            self._parent = parent_actor
            
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
        #lane cross
        print('Crossed line %s' % ' and '.join(text))
        print(text[0])
        self.text = text[0]
        font = get_font0()
        if text[0] == 'SolidSolid' or (len(text[0]) == 10 and text[0].find('S') != -1):
            self.doubleline = True
        self.display.blit(font.render('Crossed line %s' % ' and '.join(text), True, (255, 255, 255)),(8, 82))
    
    def get_doubleline(self):
        return self.doubleline

    def get_textline(self):
        return self.text

    def reset_doubleline(self):
        self.doubleline = False
    def reset_textline(self):
        self.text = ""


class CarlaSyncMode(object):


    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data

def depthcal(location_array,dep_array):
    depth = 1
    (left, right, top, bottom) = (int(location_array[1]*800), int(location_array[3]*800),int(location_array[0]*600),int(location_array[2]*600))
    color1 = dep_array[left, bottom]
    color2 = dep_array[right, bottom]
    normalized1 = (color1[0] + color1[1] * 256 + color1[2] * 256 * 256) / (256 * 256 * 256 - 1)
    depth1 = normalized1 * 1000
    normalized2 = (color2[0] + color2[1] * 256 + color2[2] * 256 * 256) / (256 * 256 * 256 - 1)
    depth2 = normalized2 * 1000
    if (left > 400):
        depth = depth1
    if (right < 400):
        depth = depth2
    return depth

def depthcal2(location_array,dep_array):
    depth = []
    draw_location = []

    for arr in location_array:
        (left, right, top, bottom) = (int(arr[1]*800), int(arr[3]*800),int(arr[0]*600),int(arr[2]*600))
        if left == 800:
            left -= 1
        if right == 800:
            right -= 1
        if top == 600:
            top -= 1
        if bottom == 600:
            bottom -= 1
        color1 = dep_array[bottom, left]
        color2 = dep_array[bottom, right]
        normalized1 = (color1[0] + color1[1] * 256 + color1[2] * 256 * 256) / (256 * 256 * 256 - 1)
        depth1 = normalized1 * 1000
        normalized2 = (color2[0] + color2[1] * 256 + color2[2] * 256 * 256) / (256 * 256 * 256 - 1)
        depth2 = normalized2 * 1000
        if (left > 400):
            depth.append(depth1)
        else:
            depth.append(depth2)
        center = [(left+right)/2,(top+bottom)/2]
        draw_location.append(center)
    return depth, draw_location

def depthcalLowerQuatile(location_array,dep_array): #classes, score, 
    depthmin = []
    depth1q = []
    depth3q = []
    draw_location = []
    CONST_REPEAT = 5
    for arr in location_array:
        hori_loop = []
        vert_loop = []
        (left, right, top, bottom) = (int(arr[1]*800), int(arr[3]*800),int(arr[0]*600),int(arr[2]*600))
        if left == 800:
            left -= 1
        if left == 800:
            right -= 1
        if top == 600:
            top -= 1
        if bottom == 600:
            bottom -= 1
        batch_hori = int((right - left)/CONST_REPEAT)
        if batch_hori > 0:
            hori_loop = np.arange(left,right,batch_hori)
        #print(hori_loop)

        """ if (hori_loop[-1] >= 800):
            print('************************',hori_loop[-1])
            hori_loop[-1] = 799 """
        #batches_hori = [left,right,right]
        batch_vert = int((bottom - top)/CONST_REPEAT)
        if batch_vert > 0:
            vert_loop = np.arange(top,bottom,batch_vert)
        #print(vert_loop)
        """ if (vert_loop[-1] >= 600):
            print('########################',vert_loop[-1])
            vert_loop[-1] = 599 """
        depth_list = []
        if len(hori_loop) == 0 or len(vert_loop) ==0:
            continue
        x_starter , y_starter = left, top
        for i in hori_loop:
            for j in vert_loop:
                
                color1 = dep_array[j, i]
                #y_starter += batch_hori
                normalized1 = (color1[0] + color1[1] * 256 + color1[2] * 256 * 256) / (256 * 256 * 256 - 1)
                depth1 = normalized1 * 1000
                depth_list.append(depth1)
                '''color1 = dep_array[bottom, left]
                color2 = dep_array[bottom, right]
                normalized1 = (color1[0] + color1[1] * 256 + color1[2] * 256 * 256) / (256 * 256 * 256 - 1)
                depth1 = normalized1 * 1000
                normalized2 = (color2[0] + color2[1] * 256 + color2[2] * 256 * 256) / (256 * 256 * 256 - 1)
                depth2 = normalized2 * 1000'''
            #x_starter += batch_vert
        
        depth_list = sorted(depth_list)

        quantile = np.percentile(depth_list,[0.25,0.5,0.75])
        #high quatile
        depthmin.append(depth_list[0])
        depth1q.append(quantile[0])
        depth3q.append(quantile[2])
        center = [(left+right)/2,(top+bottom)/2]
        draw_location.append(center)
    return depthmin, depth1q, depth3q, draw_location
    

def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)
def get_font2(size):
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, int(size))


def loadmodel():
    print('Loading model...')
    PATH_TO_SAVED_MODEL = MODEL_PATH + "/saved_model"

    start_time = time.time()
    print(start_time)
    # Load saved model and build the detection function
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))
    return detect_fn

def draw_image(surface, image,image_depth, model,category_index, blend, camera_constant):
    #camera_constant = round(camera_constant,1)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    """ input_tensor = tf.convert_to_tensor(array)
    input_tensor = input_tensor[tf.newaxis, ...] """
    """ print("shape image array")
    print(array.shape)
    print("image ")
    print(image) """
    array2 = np.frombuffer(image_depth.raw_data, dtype=np.dtype("uint8"))
    array2 = np.reshape(array2, (image_depth.height, image_depth.width, 4))
    array2 = array2[:, :, :3]
    array2 = array2[:, :, ::-1]
    """ print("shape imagedep array")
    print(array2.shape)
    print("imagedep ")
    print(image_depth) """
    input_tensor = tf.convert_to_tensor(array)
    input_tensor = input_tensor[tf.newaxis, ...]
    print('Predicting...')
    start_time = time.time()
    ######################
    model_fn = model.signatures['serving_default']

    detections = model_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}


    #####################
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))

    
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    """ print(detections['detection_boxes'])
    print(detections['detection_classes']) """
    #########################################################
    #depth_listmin, depth_list, depth_list3q , depth_locate_list = depthcal2(detections['detection_boxes'],array2)
    depth_listmin, depth_list, depth_list3q , depth_locate_list = depthcalLowerQuatile(detections['detection_boxes'],array2)
    
    #depth_list , depth_locate_list = depthcal2(detections['detection_classes'],detections['detection_scores'],detections['detection_boxes'],array2)
    #print("depth_list:{}, depth_locate_list:{}".format(len(depth_list),len(depth_locate_list)))
    """ print(depth_list)
    print(depth_locate_list) """
    #########################################################
    
    #############################
    
    ##############################
    image_np_with_detections = array.copy()
    #print(image_np_with_detections)
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)

    image_surface = pygame.surfarray.make_surface(image_np_with_detections.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))
    font = get_font2(20)
    #norm = [(1-float(i)/max(depth_list)) for i in depth_list]
    """ for i in range(len(depth_list)):
        #font = get_font2(40*norm[i])
        g , b = 255 , 255
        if depth_list[i]<25 and depth_locate_list[i][1]>300:
            g , b = 255*(depth_list[i]/30) , 255*(depth_list[i]/30)
        text_surface = font.render(str(int(depth_list[i])), True, (255, g, b))
        surface.blit(text_surface, (depth_locate_list[i][0], depth_locate_list[i][1])) """

    for i in range(len(depth_list)):
        #font = get_font2(40*norm[i])
        g , b = 255 , 255
        if depth_list[i]<25 and depth_locate_list[i][1]>300:
            g , b = 255*(depth_list[i]/30) , 255*(depth_list[i]/30)
        #textrange = "Min:{}, LQ:{}, HQ:{}".format(str(round(depth_listmin[i]-camera_constant,1)),str(round(depth_list[i]-camera_constant,1)),str(round(depth_list3q[i]-camera_constant,1)))
        textrange = "LQ:{}".format(str(round(depth_list[i]-camera_constant,1)))
        text_surface = font.render(textrange, True, (255, g, b))
        surface.blit(text_surface, (depth_locate_list[i][0], depth_locate_list[i][1]))

    if len(depth_list) > 0:
        if min(depth_list)<8:
            font = get_font2(30)
            text = font.render("Caution!!!!!", True, (255, 255, 255))
            text_rect = text.get_rect(center=(800/2, 18))
            temp_surface = pygame.Surface(text.get_size())
            temp_surface.fill((255, 0, 0))
            temp_surface.blit(text, (0, 0))
            surface.blit(temp_surface, text_rect)
    
    return depth_listmin, depth_list, depth_list3q, detections['detection_classes']


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False
##################################################



def main():
    actor_list = []
    model = loadmodel()
    #waitload = input("run")

    pygame.init()

    time.sleep(10)
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(2.0)

    oriworld = client.get_world()
    
    m = oriworld.get_map()
    start_pose = m.get_spawn_points()[0]
    waypoint = m.get_waypoint(start_pose.location)

    blueprint_library = oriworld.get_blueprint_library()
    vehiclebp = blueprint_library.filter("model3")[0]
    '''vehicle = oriworld.spawn_actor(
            filter("model3")[0],
            start_pose)'''
    vehicle = oriworld.try_spawn_actor(vehiclebp, start_pose)
    print(vehicle.bounding_box.extent.x)
    #print(vehicle.bounding_box.extent[0])
    camera_constant = vehicle.bounding_box.extent.x - CAMERA_DEPTH
    camera_constant = round(camera_constant,1)
    actor_list.append(vehicle)
    vehicle.set_simulate_physics(True)
    """ carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
        attach_to=vehicle) """
    camera_rgb = oriworld.spawn_actor(
        blueprint_library.find('sensor.camera.rgb'),
        #0.5
        carla.Transform(carla.Location(x=CAMERA_DEPTH, z=2.0), carla.Rotation(pitch=0)),
        attach_to=vehicle) 
    actor_list.append(camera_rgb)
    """ carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
        attach_to=vehicle) """
    camera_semseg = oriworld.spawn_actor(
        blueprint_library.find('sensor.camera.depth'),
        #2.8
        carla.Transform(carla.Location(x=CAMERA_DEPTH, z=2.0), carla.Rotation(pitch=0)),
        attach_to=vehicle)
    actor_list.append(camera_semseg)

    world = World(oriworld,vehicle)
    controller = KeyboardControl(world, False)
    #lane_invasion_sensor = LaneInvasionSensor(vehicle)


    display = pygame.display.set_mode(
        (800, 600),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()
    
    lane_invasion_sensor = LaneInvasionSensor(vehicle,display)
    counter = 0
    
    category_index = label_map_util.create_category_index_from_labelmap(LABEL_PATH,
                                                                    use_display_name=True)
    try:
        # Create a synchronous mode context.
        with CarlaSyncMode(oriworld, camera_rgb, camera_semseg, fps=5) as sync_mode:
            while True:
                if should_quit():
                    return
                clock.tick()
                warning = False
                warningMessage = ""
                # Advance the simulation and wait for the data.
                snapshot, image_rgb, image_semseg = sync_mode.tick(timeout=1.5)
                counter += 1
                if counter == 10:
                    counter = 0
                """ image_semseg.save_to_disk('_out/depe%06d.jpg') """
                # Choose the next waypoint and update the car location.
                #waypoint = random.choice(waypoint.next(1.5))
                #vehicle.set_transform(waypoint.transform)

                """ image_semseg.convert(carla.ColorConverter.CityScapesPalette) """
                fps = round(1.0 / snapshot.timestamp.delta_seconds)
                if controller.parse_events(client, world, clock, True):
                    return
                
                # Draw the display.
                depth_listmin, depth_list, depth_list3q, objects = draw_image(display, image_rgb, image_semseg, model, category_index,False, camera_constant)
                print("get_doubleline",lane_invasion_sensor.get_doubleline())
                print("get_textline",lane_invasion_sensor.get_textline())
                crossingline = lane_invasion_sensor.get_textline()
                #if crossingline == "SolidSolid":
                if lane_invasion_sensor.get_doubleline():
                    print('soliddoubleline')
                    warning = True
                    warningMessage += "Crossing Double white line; "
                elif len(crossingline) == 10:
                    print('soliddoubleline')
                    warning = True
                    warningMessage += "Crossing Double white line; "
                lane_invasion_sensor.reset_doubleline()
                lane_invasion_sensor.reset_textline()
                if len(depth_list) == len(objects):
                    print('same distance')
                dictkeylist =[]
                dictvaluelist = []
                for filter in range(len(objects)):
                    #print('objects[filter] ',objects[filter])
                    if (objects[filter] in important_list):
                        dictvaluelist.append(depth_list[filter])
                        dictkeylist.append(classes[objects[filter]])

                dictOfObjectDistance = dict(zip(dictvaluelist, dictkeylist))

                timestring = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                

                #stat
                t = world.player.get_transform()
                v = world.player.get_velocity()
                #'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

                speed = (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
                if speed > 70:
                    warning = True
                    warningMessage += "Speed over 70km/h; "
                if len(dictvaluelist) > 0:
                    print("mindis",min(dictvaluelist))
                    if (speed > 10) and min(dictvaluelist) <=2:
                        warning = True
                        warningMessage += "Distance from other objects are too close during driving(10KM/H & 2M); "
                if len(dictvaluelist) > 0:
                    print("mindis",min(dictvaluelist))
                    if (speed > 50) and min(dictvaluelist) <=5:
                        warning = True
                        warningMessage += "Distance from other objects are too close during driving(50KM/H & 5M); "
                    
                virtual_location = [round(t.location.x,1), round(t.location.y,1)]

                

                #'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y))
                """ draw_image(display, image_semseg, blend=True) """
                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                    (8, 28))
                display.blit(
                    font.render('Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)), True, (255, 255, 255)),
                    (8, 46))
                display.blit(
                    font.render('Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)), True, (255, 255, 255)),
                    (8, 64))
                pygame.display.flip()
                
                #DRIVER_ID, DISPALY, speed, CROSSING, location, distance_set, distance_classes, warning, warningMessage, timestring, time.time()
                #pygame.image.save()
                address = 'C:\\Users\\Acer\\Downloads\\CARLA_0.9.12\\WindowsNoEditor\\PythonAPI\\examples\\carcam_image\\'
                jpgtimestring = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
                filename = address+jpgtimestring+"("+str(counter)+").jpg"
                newImageRecord = ImageRecord(DRIVER_ID, display, speed, crossingline, virtual_location, dictvaluelist, dictkeylist, warning, warningMessage, timestring, time.time())
                if warning:
                    pygame.image.save(display,filename)
                    r = requests.post('http://localhost:3001/carEvents',json=newImageRecord.statusReturn())
                    response = r.json()
                    print(response)

                    files =  open(filename, 'rb')
                    my_files = {'file':files}
                    r2 = requests.post('http://localhost:3001/imagecar/'+response['id'],files=my_files)
                    response2 = r2.json()
                    print(response2)

    finally:

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print('done.')


if __name__ == '__main__':

    try:

        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
