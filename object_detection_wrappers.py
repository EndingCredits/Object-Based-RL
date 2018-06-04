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


class GenericObjectDetectionWrapper(Wrapper):
    '''
    Template for generic object detection
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
        elif type(self.base_env) is AtariEnv:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
        
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
        
        
    def _draw_bb(self, bounding_boxes, draw_ground_truth=True):
        if type(self.base_env) is VGDLEnv:
        
            # Draw screen (again) - not needed
            #self.base_env.game._drawAll()
            
            # Scale display
            pygame.transform.scale(self.base_env.screen,
                self.base_env.display_size, self.base_env.display)
            
            # Draw bounding boxes
            scale_size = self.base_env.zoom
            
            def draw_boxes(boxes, color, thickness=1):
                for box in boxes:
                    x = box[0] * scale_size
                    y = box[1] * scale_size
                    width = box[2] * scale_size
                    height = box[3] * scale_size
                    pygame.draw.rect(self.base_env.display,
                        color, (x, y, width, height), thickness)
            
            # Ground truth
            if draw_ground_truth:
                true_boxes = self.base_env.game.getBoundingBoxes()  
                draw_boxes(true_boxes, (255,0,0), 1)
                text = self.font.render(str(len(true_boxes)), True, (255,0,0))
                self.base_env.display.blit(text, text.get_rect().move(0,30))
                
            # Detected truth
            draw_boxes(bounding_boxes, (0,255,0), 2)
                           
            
            #Display text
            text = self.font.render(str(len(bounding_boxes)), True, (0,255,0))
            self.base_env.display.blit(text, text.get_rect())
            
            pygame.display.update()
            
        elif type(self.base_env) is AtariEnv:
            #TODO: implement for ALE
            img = self.base_env._get_image()
            
            def draw_boxes(boxes, color, thickness=1):
                c = np.array(color)
                for box in boxes:
                    img[box[1], box[0]:box[0] + box[2]] = c
                    img[box[1] + box[3]-1, box[0]:box[0] + box[2]] = c
                    img[box[1]:box[1] + box[3], box[0]] = c
                    img[box[1]:box[1] + box[3], box[0] + box[2] -1] = c
                    
            # Detected truth
            draw_boxes(bounding_boxes, (0,255,0), 2)
            
            self.viewer.imshow(img)
        
        
        
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

        boxes = []
        for idx, cont in enumerate(approx):
            x,y,w,h = cv2.boundingRect(cont)
            boxes.append([x,y,w,h])
            
        return boxes
        
        
    def _get_object_features(self, box):
        return np.array(box)
