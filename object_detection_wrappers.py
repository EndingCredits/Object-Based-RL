from gym import Wrapper
from gym import spaces

from gym_vgdl import VGDLEnv
from gym_vgdl.list_space import list_space
import pygame
from pygame.locals import *

from gym_utils.frame_history_wrapper import FrameHistoryWrapper

from copy import deepcopy

import cv2
import numpy as np


class GenericObjectDetectionWrapper(Wrapper):
    '''
    Template for
    '''
    
    # Stores last 2 frames
    history_length = 2
    
    # Total number of object attributes inc position etc
    num_attributes = 10
    
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
        
    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames = obs
        
        return self._get_state(), reward, done, info

    def _reset(self):
        self.frames = self.env.reset()
        return self._get_state()

    def _get_state(self):
        # Get boxes
        boxes = self._detect_bb(self.frames)
        
        object_list = []
        for box in boxes:
           object_list.append(self._get_object_features(box))
        
        self._draw_bb(boxes)
        
        return object_list if object_list else [np.zeros(self.num_attributes)]
        
        
    def _draw_bb(self, bounding_boxes):
        if type(self.base_env) is VGDLEnv:
        
            # Draw screen (again) - not needed
            #self.base_env.game._drawAll()
            
            # Scale display
            pygame.transform.scale(self.base_env.screen,
                self.base_env.display_size, self.base_env.display)
            
            # Draw bounding boxes
            scale_size = self.base_env.zoom
            for box in bounding_boxes:
                x = box[0] * scale_size
                y = box[1] * scale_size
                width = box[2] * scale_size
                height = box[3] * scale_size
                thickness = 2
                color = (255,0,0)
                pygame.draw.rect(self.base_env.display,
                    color, (x, y, width, height), thickness)
            
            #Display text
            text = self.font.render(str(len(bounding_boxes)), True, (0, 128, 0))
            self.base_env.display.blit(text, text.get_rect())
            
            pygame.display.update()
            
        #TODO: implement for ALE
        
        
        
    def _detect_bb(self, frames):
        # Should return a list, where each element is of the form
        # [  x, y, w, h, ... ]
        
        bounding_boxes = []
        
        return bounding_boxes
        
        
    def _get_object_features(self, box):
        return np.zeros(self.num_attributes)
        
        
    def render(self):
        #TODO: handle this more gracefully
        pass
    
    
    
    
    
from feature_extractors import OpenCVBoundingBoxExtractor

class TestObjectDetectionWrapper(GenericObjectDetectionWrapper):

    # Stores last 2 frames
    history_length = 1
    
    # Total number of object attributes inc position etc
    num_attributes = 4
    
    ARC_THRESHOLD = 10
    
    def __init__(self, env):
        assert env is not None
        super(TestObjectDetectionWrapper, self).__init__(env)
        
 
    def _detect_bb(self, frames):
    
        # Taken from https://github.com/wulfebw/playing_atari/blob/master/scripts/common/feature_extractors.py
        img = deepcopy(frames[-1]) 
        imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(imgray, 20, 100)
        ret, thresh = cv2.threshold(edges, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, 
            cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = [cont for cont in contours
            if cv2.arcLength(cont, True) > self.ARC_THRESHOLD]

        approx = []
        for cnt in contours:
            epsilon = 0.00*cv2.arcLength(cnt,True)
            approx.append(cv2.approxPolyDP(cnt,epsilon,True))

        boxes = []
        for idx, cont in enumerate(approx):
            x,y,w,h = cv2.boundingRect(cont)
            boxes.append([x,y,w,h])
            
        return boxes
        
        
    def _get_object_features(self, box):
        return np.array(box)
