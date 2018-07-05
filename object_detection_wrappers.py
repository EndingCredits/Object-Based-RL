from gym import Wrapper
from gym import spaces

from gym.envs.atari.atari_env import AtariEnv
from gym_vgdl import VGDLEnv
from gym_vgdl.list_space import list_space
import pygame
from pygame.locals import *

try:
    from gym_utils.frame_history_wrapper import FrameHistoryWrapper
except:
    from frame_history_wrapper import FrameHistoryWrapper

from copy import deepcopy

import cv2
import numpy as np
from scipy.spatial import distance
from sklearn.cluster import DBSCAN


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
        elif type(self.base_env) is AtariEnv:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()

        self.prev_frame_objects = []
        self.object_class_examples = []
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
        self._draw_bb(objects)

        return objects if objects else [np.zeros(self.num_attributes)]


    def _draw_bb(self, objects, draw_ground_truth=True):
        if len(self.colors) < len(self.object_class_examples):
            for idx in range(len(self.colors),len(self.object_class_examples)):
                self.colors.append(list(np.random.choice(range(256), size=3)))

        if type(self.base_env) is VGDLEnv:

            # Scale display
            pygame.transform.scale(self.base_env.screen,
                self.base_env.display_size, self.base_env.display)

            # Draw bounding boxes
            scale_size = self.base_env.zoom

            def draw_boxes(boxes, box_colors, thickness=1):
                for ind, box in enumerate(boxes):
                    x = box[0] * scale_size
                    y = box[1] * scale_size
                    width = box[2] * scale_size
                    height = box[3] * scale_size
                    pygame.draw.rect(self.base_env.display,
                        box_colors[ind], (x, y, width, height), thickness)

            # Ground truth
            if draw_ground_truth:
                true_boxes = self.base_env.game.getBoundingBoxes()
                box_colors = [ (255,0,0) for _ in range(len(true_boxes)) ]
                draw_boxes(true_boxes, box_colors, 1)
                text = self.font.render(str(len(true_boxes)), True, (255,0,0))
                self.base_env.display.blit(text, text.get_rect().move(0,30))

            # Detected truth
            bounding_boxes = [ obj[0:4] for obj in objects ]
            box_colors = [ self.colors[obj[4]] for obj in objects ]
            draw_boxes(bounding_boxes, box_colors, 2)

            #Display text
            text = self.font.render(str(len(objects)), True, (0,255,0))
            self.base_env.display.blit(text, text.get_rect())

            pygame.display.update()

        elif type(self.base_env) is AtariEnv:
            #TODO: implement for ALE
            img = self.base_env._get_image()

            def draw_boxes(boxes, box_colors, thickness=1):
                for idx, box in enumerate(boxes):
                    c = np.array[box_colors[idx]]
                    img[box[1], box[0]:box[0] + box[2]] = c
                    img[box[1] + box[3]-1, box[0]:box[0] + box[2]] = c
                    img[box[1]:box[1] + box[3], box[0]] = c
                    img[box[1]:box[1] + box[3], box[0] + box[2] -1] = c

            # Detected truth
            bounding_boxes = [ obj[0:4] for obj in objects ]
            box_colors = [ self.colors[obj[4]] for obj in objects ]
            draw_boxes(bounding_boxes, box_colors, 2)

            self.viewer.imshow(img)


    def _detect_objects(self, frame):
        # Should return a list, where each element is of the form
        # [  x, y, w, h, ... ]
        objects = []
        return objects


    def render(self):
        #TODO: handle this more gracefully
        pass



class TestObjectDetectionWrapper(GenericObjectDetectionWrapper):

    # Stores last 2 frames
    history_length = 2

    # Total number of object attributes inc position etc
    num_attributes = 6

    ARC_THRESHOLD = 10

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


        cont_classes_labels = self._cluster_contours(contours, img)
        objects = []
        for idx, cont in enumerate(approx):
            x,y,w,h = cv2.boundingRect(cont)
            objclass = cont_classes_labels[idx]
            velocity = self._get_object_velocity(cont, objclass)
            objects.append([x, y, w, h, objclass, velocity])

        return objects


    def _cluster_contours(self, contours, img):
        objs = []
        for cont in contours:
            x,y,w,h = cv2.boundingRect(cont)
            mask = np.zeros(img.shape[:2], dtype="uint8")
            cv2.drawContours(mask, [cont], -1, 255, -1)
            mask = cv2.erode(mask, None, iterations=2)
            mean_colour = cv2.mean(img, mask=mask)[:3]
            objs.append([w, h] + list(mean_colour)) # need to normalise the values

        objs = [y for x in self.object_class_examples for y in x] + objs # add previous objects to ensure consistent classes
        labels = DBSCAN(min_samples=1, eps=10).fit_predict(objs)
        self._store_objects_by_class(objs, labels)

        return labels[-len(contours):] # get the labels only of the current contours


    def _store_objects_by_class(self, objs, labels):
        objs_sorted = [x for _,x in sorted(zip(labels,objs))]
        labels_sorted = sorted(labels)
        obj_cl = []

        for idx, obj in enumerate(objs_sorted):
            if len(self.object_class_examples) <= labels_sorted[idx]:
                self.object_class_examples.append([obj])
            else:
                self.object_class_examples[labels_sorted[idx]].append(obj)
        self.object_class_examples = [examples[-5:] for examples in
                                      self.object_class_examples]


    def _get_object_velocity(self, contour, objclass):
        x,y,w,h = cv2.boundingRect(contour)

        # get closest boxes from last frame
        distances_between = [ distance.euclidean((obj[0], obj[1]), (x,y))
                             for obj in self.prev_frame_objects
                             if objclass == obj[4] ]

        # velocity is calculated as the difference between this x,y and last frames x,y
        return min(distances_between) if len(distances_between) != 0 else 0


    def _get_object_features(self, box, boxes): # box = [x, y, width, height]
        return np.array(box)
