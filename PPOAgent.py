"""
Adapted from NVIDIA GA3C code
"""

from __future__ import division

import numpy as np
import cv2
import tensorflow as tf


from ops import linear

from Actor import RolloutActor

from threading import Thread
from multiprocessing import Process, Queue, Value

import time

class ActorProcess(Process, RolloutActor):
    def __init__(self,
                 id,
                 prediction_queue,
                 env_constructor,
                 training_queue,
                 state_seq_len=1):
        Process.__init__(self)
        RolloutActor.__init__(self)
        
        self.id = id
        self.prediction_output = prediction_queue
        self.prediction_input = Queue(maxsize=1)
        self.training_queue = training_queue
        self.exit_flag = Value('i', 0)
        
        self.env_constructor = env_constructor

    def _act(self, state):
        self.prediction_output.put((self.id, state))
        act, other = self.prediction_input.get()
        return act, other
        
    def _save_trajectory(self):
        self.training_queue.put(self.trajectory)

    def get_prediction_output(self):
        return self.prediction_output
        
    def run(self):
        env = self.env_constructor()
        obs = env.reset()
        self.Reset(obs)

        while self.exit_flag.value == 0:
            act = self.GetAction()
            obs, reward, terminal, _ = env.step(act)
            env.render()
            self.Update(act, reward, obs, terminal)
            
            if terminal:
                obs = env.reset()
                self.Reset(obs)


class PredictorThread(Thread):
    def __init__(self,
                 id,
                 input_queue,
                 output_queues,
                 prediction_model ):
        super(PredictorThread, self).__init__()
        self.setDaemon(True)
        self.id = id
        self.input_queue = Queue()
        self.output_queues = output_queues
        self.model = prediction_model
        self.exit_flag = False
        
        
    def run(self):
        batch_size = 16
        
        ids = [ None ] * batch_size
        states = [ None ] * batch_size

        while not self.exit_flag:
            size = 0
            while size < batch_size and not self.input_queue.empty():
                ids[size], states[size] = self.input_queue.get()
                size += 1
                
            if size != 0:
                batch = states[:size]
                p, v = self.model._get_action(batch)

                for i in range(size):
                  if i < len(self.output_queues):
                    self.output_queues[ids[i]].put((p[i], v[i]))
                
                             
class PPOAgent(object):
    def __init__(self, session, env_constructor, args):
         
        # Set up agent parameters:
        ##################################

        self.name = "PPOAgent"
        
        # Environment details
        self.obs_size = args.obs_size
        self.n_actions = args.num_actions
        self.viewer = None

        # Reinforcement learning parameters
        self.discount = args.discount
        #self.n_steps = args.n_step
        self.initial_epsilon = args.epsilon
        self.epsilon = self.initial_epsilon
        self.epsilon_final = args.epsilon_final
        self.epsilon_anneal = args.epsilon_anneal

        # Training parameters
        self.model_type = args.model
        self.history_len = args.history_len
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        
        # Set up other variables:
        ##################################

        # Running variables
        self.train_step = 0
        self.seed = args.seed
        self.rng = np.random.RandomState(self.seed)
        self.session = session
        
        self.num_actors = 0
        self.actors = []
        
        self.env = env_constructor
        
        # Thread queues
        self.prediction_queue = Queue()
        self.training_queue = Queue()
        
    def _get_action(self, state):
        num_states = len(state)
        return [1]*num_states, [1]*num_states
        
    def _add_actor(self):
        actor = ActorProcess(self.num_actors,
                             self.prediction_queue,
                             self.env,
                             self.training_queue,
                             self.history_len )
        self.actors.append(actor)
        self.actors[-1].start()
        self.num_actors += 1
    
    def _remove_actor(self):
      if self.num_actors > 0:
        self.num_actors -= 1
        self.actors[-1].exit_flag.value = True
        self.actors[-1].join()
        self.actors.pop()
        
    def _update_actors(self, batch_size=32):
        for actor in self.actors:
            size = 0
            ids = [ None ] * batch_size
            states = [ None ] * batch_size
            
            # Get 
            while size < batch_size and not self.prediction_queue.empty():
                ids[size], states[size] = self.prediction_queue.get()
                size += 1
                
            if size != 0:
                batch = states[:size]
                p, v = self._get_action(batch)

                for i in range(size):
                    self.actors[ids[i]].prediction_input.put((p[i], v[i]))
                    
    def _get_training_batch(self, batch_size=32):
        pass
        
        

    def Update(self):
        self._update_actors()
    


def batch_objects(input_list):
    # Takes an input list of lists (of vectors), pads each list the length of
    #   the longest list, compiles the list into a single n x m x d array, and
    #   returns a corresponding n x m x 1 mask.
    max_len = 0
    out = []; masks = []
    for i in input_list:
        max_len = max(len(i),max_len)
    for l in input_list:
        # Zero pad output
        out.append(np.pad(np.array(l,dtype=np.float32),
            ((0,max_len-len(l)),(0,0)), mode='constant'))
    return out


if __name__ == '__main__':

    import argparse
    import gym
    import gym_vgdl
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0',
                       help='Gym environment to use')
    parser.add_argument('--model', type=str, default=None,
                       help='Leave None to automatically detect')
    parser.add_argument('--objdetect', type=int, default=0,
                       help='Set true to use object detection and object detection network')

    parser.add_argument('--seed', type=int, default=123,
                       help='Seed to initialise the agent with')
                       
    parser.add_argument('--render', type=int, default=1,
                       help='Set false to diable rendering')

    parser.add_argument('--training_iters', type=int, default=5000000,
                       help='Number of training iterations to run for')
    parser.add_argument('--display_step', type=int, default=25000,
                       help='Number of iterations between parameter prints')
    parser.add_argument('--test_step', type=int, default=50000,
                       help='Number of iterations between tests')
    parser.add_argument('--test_count', type=int, default=5,
                       help='Number of test episodes per test')

    parser.add_argument('--learning_rate', type=float, default=0.00025,
                       help='Learning rate for TD updates')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Size of batch for Q-value updates')
    parser.add_argument('--replay_memory_size', type=int, default=100000,
                       help='Size of replay memory')
    parser.add_argument('--learn_step', type=int, default=4,
                       help='Number of steps in between learning updates')

    parser.add_argument('--discount', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='Initial epsilon')
    parser.add_argument('--epsilon_final', type=float, default=None,
                       help='Final epsilon')
    parser.add_argument('--epsilon_anneal', type=int, default=None,
                       help='Epsilon anneal steps')

    parser.add_argument('--save_file', type=str, default=None,
                       help='Name of save file for test results (leave None for no saving)')                 
    parser.add_argument('--chk_file', type=str, default=None,
                       help='Name of save file (leave None for no saving)')
                       
    args = parser.parse_args()
    
    arg_dict = vars(args)
    col_a_width = 20 ; col_b_width = 16
    print(' ' + '_'*(col_a_width+1+col_b_width) + ' ')
    print('|' + ' '*col_a_width + '|' + ' '*col_b_width  + '|')
    line = "|{!s:>" + str(col_a_width-1) + "} | {!s:<" + str(col_b_width-1) + "}|"
    for i in arg_dict:
        print(line.format(i, arg_dict[i]))
    print('|' + '_'*col_a_width + '|' + '_'*col_b_width  + '|')
    print('')
    
       
    env_cons = lambda: gym.make(args.env)
    
    env = env_cons()
    args.num_actions = env.action_space.n
    args.obs_size = list(env.observation_space.shape)
    args.history_len = 0
    env.close()
    
    agent = PPOAgent(None, env_cons, args)
    
    for i in range(10):
        agent._add_actor()
    for i in range(1000000):
        agent.Update()

