from gym import Wrapper
from gym import spaces

from gym.envs.atari.atari_env import AtariEnv
from gym_vgdl import VGDLEnv
from gym_vgdl.list_space import list_space
import pygame
from pygame.locals import *

from gym_utils.frame_history_wrapper import FrameHistoryWrapper

from copy import deepcopy

import cv2
import numpy as np
from scipy.spatial import distance


class GenericObjectDetectionWrapper(Wrapper):
    '''
    Template for generic object detection
    '''

    # Stores last 2 frames
    history_length = 2

    # Total number of object attributes inc position etc
    num_attributes = 6


    def __init__(self, env):
        assert env is not None

        # First wrap input with Frame history
        env = FrameHistoryWrapper(env, self.history_length)

        # Initialise
        super(GenericObjectDetectionWrapper, self).__init__(env)

        # New observation space for object representation
        self.observation_space = list_space( spaces.Box(
            low=0, high=255, shape=[self.num_attributes]) )

        # Get the base environment (just saves some calls)
        self.base_env = self.unwrapped

        if type(self.base_env) is VGDLEnv:
            pygame.font.init()
            self.font = pygame.font.SysFont(None, 48)
        self.viewer = None


        self.prev_frame_objects = []
        self.colors = []


    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames = obs
        return self._get_state(), reward, done, info


    def _reset(self):
        self.frames = self.env.reset()
        return self._get_state()


    def _get_state(self):
        objects = self._detect_objects(self.frames[-1])
        self.prev_frame_objects = objects

        return objects if objects else [np.zeros(self.num_attributes)]


    def _draw_bb(self, objects, draw_ground_truth=True, img_only=False):
        if type(self.base_env) is VGDLEnv:

            # Scale display
            display = pygame.display.get_surface()
            pygame.transform.scale(self.base_env.screen,
                self.base_env.display_size, display)

            # Draw bounding boxes
            scale_size = self.base_env.zoom

            def draw_boxes(boxes, box_colors, thickness=1):
                for ind, box in enumerate(boxes):
                    x = box[0] * scale_size
                    y = box[1] * scale_size
                    width = box[2] * scale_size
                    height = box[3] * scale_size
                    pygame.draw.rect(display,
                        box_colors[ind], (x, y, width, height), thickness)

            # Ground truth
            if draw_ground_truth:
                true_boxes = self.base_env.game.getBoundingBoxes()
                box_colors = [ (255,0,0) for _ in range(len(true_boxes)) ]
                draw_boxes(true_boxes, box_colors, 1)
                text = self.font.render(str(len(true_boxes)), True, (255,0,0))
                display.blit(text, text.get_rect().move(0,30))

            # Detected truth
            bounding_boxes = [ obj[0:4] for obj in objects ]
            box_colors = [ (0,255,0) for _ in range(len(objects)) ]
            draw_boxes(bounding_boxes, box_colors, 2)

            #Display text
            text = self.font.render(str(len(objects)), True, (0,255,0))
            display.blit(text, text.get_rect())

            pygame.display.update()
            if img_only:
                return np.flipud(np.rot90(pygame.surfarray.array3d(
                   display).astype(np.uint8)))


        elif type(self.base_env) is AtariEnv:
            #TODO: implement for ALE
            img = self.base_env._get_image()

            def draw_boxes(boxes, box_colors, thickness=1):
                for idx, box in enumerate(boxes):
                    c = np.array(box_colors[idx])
                    img[box[1], box[0]:box[0] + box[2]] = c
                    img[box[1] + box[3]-1, box[0]:box[0] + box[2]] = c
                    img[box[1]:box[1] + box[3], box[0]] = c
                    img[box[1]:box[1] + box[3], box[0] + box[2] -1] = c

            # Detected truth
            bounding_boxes = [ obj[0:4] for obj in objects ]
            box_colors = [ (0,255,0) for _ in range(len(objects)) ]
            draw_boxes(bounding_boxes, box_colors, 2)

            if img_only:
                return img
            else:
                if self.viewer is None:
                    from gym.envs.classic_control import rendering
                    self.viewer = rendering.SimpleImageViewer()
                self.viewer.imshow(img)


    def _detect_objects(self, frame):
        # Should return a list, where each element is of the form
        # [  x, y, w, h, ... ]
        objects = []
        return objects


    def render(self, mode='human', close=False):
        if mode == 'rgb_array':
            return self._draw_bb(self.prev_frame_objects, img_only=True)
        else:
            self._draw_bb(self.prev_frame_objects)
            return True



class TestObjectDetectionWrapper(GenericObjectDetectionWrapper):

    # Stores last 2 frames
    history_length = 1

    # Total number of object attributes inc position etc
    num_attributes = 9

    ARC_THRESHOLD = 10

    OBJECT_MATCH_MAX_DISTANCES = [1, 1, *([5]*3)]


    def __init__(self, env):
        assert env is not None
        super(TestObjectDetectionWrapper, self).__init__(env)


    def _detect_objects(self, frame):
        # Taken from https://github.com/wulfebw/playing_atari/blob/master/scripts/common/feature_extractors.py
        img = deepcopy(frame)
        imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(imgray, 20, 100)
        ret, thresh = cv2.threshold(edges, 127, 255, 0)
        try:
            contours, hierarchy = cv2.findContours(thresh,
                cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        except:
            _, contours, hierarchy = cv2.findContours(thresh,
                cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = [cont for cont in contours
            if cv2.arcLength(cont, True) > self.ARC_THRESHOLD]

        approx = []
        for cnt in contours:
            epsilon = 0.00*cv2.arcLength(cnt,True)
            approx.append(cv2.approxPolyDP(cnt,epsilon,True))

        objects = []
        for idx, cont in enumerate(approx):
            x,y,w,h = cv2.boundingRect(cont)
            colors = self._get_object_colors(cont, img)
            velocity = self._get_object_velocity(x, y, w, h, colors)
            objects.append([x, y, w, h, *colors, *velocity])

        return objects


    def _get_object_colors(self, contour, img):
        mask = np.zeros(img.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mask = cv2.erode(mask, None, iterations=2)
        return  cv2.mean(img, mask=mask)[:3]


    # get closest boxes from last frame
    def _get_object_velocity(self, x, y, w, h, colors):
        same_class_objects = [o for o in self.prev_frame_objects
                              if self._get_object_difference(o, w, h, colors) <
                              TestObjectDetectionWrapper.OBJECT_MATCH_MAX_DISTANCES ]
        if len(same_class_objects) == 0:
            return 0, 0

        min_object = min(same_class_objects,
                         key=lambda obj: distance.euclidean((obj[0], obj[1]), (x, y)))

        return min_object[0] - x, min_object[1] - y


    def _get_object_difference(self, obj, w, h, colors):
        return abs(np.array([np.array(obj[2]) - w,
                             np.array(obj[3]) - h,
                             *(np.array(obj[4:7]) - np.array(colors)) ])).tolist()


    def _get_object_features(self, box, boxes): # box = [x, y, width, height]
        return np.array(box)
