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


import cProfile, pstats


class ActorProcess(Process, RolloutActor):
    def __init__(self,
                 id,
                 prediction_queue,
                 env_constructor,
                 training_queue,
                 state_seq_len=1,
                 render=0 ):
        Process.__init__(self)
        RolloutActor.__init__(self)
        self.daemon = True
        
        self.id = id
        self.prediction_output = prediction_queue
        self.prediction_input = Queue(maxsize=1)
        self.training_queue = training_queue
        self.exit_flag = Value('i', 0)
        
        self.history_len = state_seq_len
        
        self.render_flag = Value('i', render)
        
        self.env_constructor = env_constructor

    def _act(self, state):
        self.prediction_output.put((self.id, state))
        act, probs, values = self.prediction_input.get(True, 60)
        return act, probs, values

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
            if self.render_flag.value: env.render()
            self.Update(act, reward, obs, terminal)
            
            if terminal:
                obs = env.reset()
                self.Reset(obs)


class PredictorThread(Thread):
    def __init__(self,
                 server ):
        super(PredictorThread, self).__init__(daemon=True)
        self.server = server
        self.exit_flag = False
        self.paused = False
        
        self.min_batch_size = 8
        self.max_batch_size = 32

    def run(self):
        server = self.server
        while not self.exit_flag:
          while self.paused:
            time.sleep(0.01)
          
          size = 0
          ids = [ None ] * self.max_batch_size
          states = [ None ] * self.max_batch_size
          
          while size < self.max_batch_size and \
            not server.prediction_queue.empty() or size < self.min_batch_size:
              ids[size], states[size] = server.prediction_queue.get()
              size += 1
              
          if size != 0:
              batch = states[:size]
              a, p, v = server._get_action(batch)

              for i in range(size):
                if i < len(server.actors):
                  server.actors[ids[i]].prediction_input.put((a[i], p[i], v[i]))
                
                
                             
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
        self.n_steps = args.n_step

        # Training parameters
        self.model_type = args.model
        self.history_len = args.history_len
        self.batch_size = args.batch_size
        self.train_count = 10
        self.learning_rate = args.learning_rate
        
        # Set up other variables:
        ##################################

        # Running variables
        self.train_step = 0 
        self.ep_rs = []
        self.last_update = time.time()
        self.seed = args.seed
        self.rng = np.random.RandomState(self.seed)
        self.session = session

        self.num_actors = 0
        self.actors = []
        
        self.env = env_constructor
        
        # Thread queues
        self.trajectories_queue_size = 100
        self.memory_queue_size = self.batch_size*self.train_count*100
        self.prediction_queue = Queue()
        self.training_queue = Queue(self.trajectories_queue_size)
        self.testing_queue = Queue(self.trajectories_queue_size)
        self.experience_memory = []
        
        # Predictor
        self.predictor_thread = PredictorThread( self )
        
        
        # Set up model:
        ##################################
        if self.model_type == 'CNN':
            from networks import deepmind_CNN
            self.model = deepmind_CNN
        elif self.model_type == 'fully connected':
            from networks import fully_connected_network
            self.model = fully_connected_network
        elif self.model_type == 'object':
            from networks import object_embedding_network2
            self.model = object_embedding_network2

        self._build_graph()
        self.session.run(tf.global_variables_initializer())
        
        # Start up threads:
        ##################################
        self.predictor_thread.start()
        
        
    def _build_graph(self):
    
        # Training graph
        with tf.name_scope('train'):
          with tf.device('gpu:0'):
            self._build_train_graph()
        
        # Prediction graph
        with tf.name_scope('predict'):
          with tf.device('gpu:0'):
            self._build_predict_graph()
            

    def _build_train_graph(self):
        # Build Queue pipeline
        with tf.name_scope('input_pipeline'):
            prepend = [] if self.history_len == 0 else [self.history_len]
            state_dim = prepend + self.obs_size
            
            self.model_train_queue = tf.FIFOQueue(
              self.memory_queue_size, # min_after_dequeue = 0,
              dtypes=[tf.float32, tf.int64, tf.float32, tf.float32, tf.float32],
              shapes = [ state_dim, [], [], [], [self.n_actions] ],
              names = [ 'state', 'action', 'advantage', 'return', 'actprob' ],
              name = 'model_train_queue' )
            self.model_train_queue_len = 0
            
            multi_batch_size = self.batch_size
            batch_size = tf.placeholder_with_default(multi_batch_size, [])
            batch = self.model_train_queue.dequeue_many(batch_size) 
            
            self.state = batch['state']
            self.action = batch['action']
            self.advantage = batch['advantage']
            self.returns = batch['return']
            self.old_probs = batch['actprob']
         
            feed_dict = { 'state': self.state,
                          'action': self.action,
                          'advantage': self.advantage,
                          'return': self.returns,
                          'actprob': self.old_probs,
                        }
            
            self.model_enqueue = self.model_train_queue.enqueue_many(
                                            feed_dict, name="queue_data")
        
        
        # Naive implementation of minibatch 
        
        trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                         epsilon=1e-5, name='optimizer')
        train_ops = []                                           
        for i in range(self.train_count):
          with tf.name_scope('iteration_'+str(i)):
            with tf.control_dependencies(train_ops):
              with tf.variable_scope(self.name + '_model', reuse=tf.AUTO_REUSE):
                emb = self.model(self.state)
                predict = linear(tf.nn.relu(emb), self.n_actions, name='probs')
                value = linear(tf.nn.relu(emb), 1, name='value')
              
              # From baselines code
              eps = 0.05
              vf_coef = 1.0
              ent_coef = 0.01
              
              probs = tf.nn.softmax(predict, name='normalised_probs') + 1e-5
              action_oh = tf.one_hot(self.action, self.n_actions, 1.0, 0.0)
              ratio = tf.reduce_sum(( probs / self.old_probs ) * \
                                             action_oh, axis=1, name='PPO_r')
              
              pg_loss = -self.advantage * ratio
              pg_loss_clipped = -self.advantage * tf.clip_by_value(ratio,
                                            1.0 - eps, 1.0 + eps)
              pg_loss = tf.reduce_mean(tf.maximum(pg_loss, pg_loss_clipped),
                                        name='PPO_pg_loss')
              
              vf_loss = .5 * tf.reduce_mean(tf.square(value - self.returns),
                                             name='PPO_vf_loss')
              
              entrophy = - tf.reduce_sum( probs * tf.log(probs), axis=1)
              ent_loss = tf.reduce_mean( entrophy, name='PPO_entrophy')
              
              total_loss = pg_loss - ent_loss * ent_coef + vf_loss * vf_coef
              
              train_ops.append(trainer.minimize(total_loss))
        self.train_op = tf.group(train_ops)
                  
                  
    def _build_predict_graph(self):
    
        prepend = [None] if self.history_len == 0 else [None, self.history_len]
        state_dim = prepend + self.obs_size
        
        self.state_pred = tf.placeholder("float", state_dim, name='state_ph') 
        
        with tf.variable_scope(self.name + '_model', reuse=True):
          emb = self.model(self.state_pred)
          predict = linear(tf.nn.relu(emb), self.n_actions, name='probs')
          value = linear(tf.nn.relu(emb), 1, name='value')
        
        self.action_probs = tf.nn.softmax(predict) + 1e-5
        self.value = tf.squeeze(value, axis=1)

        
    def _get_action(self, state):
        if self.obs_size[0] == None:
            state = batch_objects(state)
        probs, value = self.session.run([self.action_probs, self.value],
                                       feed_dict = { self.state_pred: state })
   
        num_act = np.shape(probs)[1]
        try:
            act = [ np.random.choice(range(num_act), p=prob ) for prob in probs ]
        except:
            print(probs)
        return act, probs, value
        
    def _add_batch(self, s, a, p, adv, ret):
        if self.obs_size[0] == None:
            s = batch_objects(s)
        feed_dict = {
            self.state: s,
            self.action: a,
            self.old_probs: p,
            self.advantage: adv,
            self.returns: ret,
        }
        self.session.run(self.model_enqueue, feed_dict=feed_dict)
        
    def _train(self, s, a, p, adv, ret):
        if self.obs_size[0] == None:
            s = batch_objects(s)
        feed_dict = {
            self.state: s,
            self.action: a,
            self.old_probs: p,
            self.advantage: adv,
            self.returns: ret,
        }
        _ = self.session.run(self.train_op, feed_dict=feed_dict)
        
        
    def _add_actor(self, rendering=0):
        actor = ActorProcess(self.num_actors,
                             self.prediction_queue,
                             self.env,
                             self.training_queue,
                             self.history_len,
                             render=rendering )
        self.actors.append(actor)
        self.actors[-1].start()
        self.num_actors += 1
    
    def _remove_actor(self):
      if self.num_actors > 0:
        self.num_actors -= 1
        self.actors[-1].exit_flag.value = True
        self.actors[-1].join()
        self.actors.pop()
                 
    def __del__(self):
        self.predictor_thread.join()
        while self.actors:
            self._remove_actor()

    def Train(self):
        while self.model_train_queue_len < self.batch_size:
            self._pump_training_queue()
        for i in range(1):
            self.model_train_queue_len -= self.batch_size
            self.session.run(self.train_op)
        
        
    def _pump_training_queue(self):  
        # Get training data
        traj = []
        while len(traj) == 0:
            traj = self.training_queue.get()
                
        self.ep_rs.append(np.sum(traj.rewards))
        
        if self.history_len == 0:
            states = traj.states
        else:
            states = [ traj.get_state(t=t, seq_len=self.history_len)
                      for t in range(len(traj)) ]
        actions = traj.actions
        #returns = np.array(traj.n_step_return(self.n_steps,
        returns = np.array(traj.gamma_return( 
                                discount=self.discount, value_index=1))
        values = np.array([ o[1] for o in traj.others ])
        probs = np.array([ o[0] for o in traj.others ])
        adv = returns - values
        
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
        
        self.model_train_queue_len += len(states)
        self._add_batch(states, actions, probs, returns, adv)
        

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
    from tqdm import tqdm, trange
    import gym
    import gym_vgdl
    
    import warnings
    warnings.filterwarnings("error", category=RuntimeWarning)
    np.seterr(all='raise')
    
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    try:
        os.environ["DISPLAY"]
    except:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    
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
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Size of batch for Q-value updates')
    parser.add_argument('--replay_memory_size', type=int, default=100000,
                       help='Size of replay memory')
    parser.add_argument('--learn_step', type=int, default=4,
                       help='Number of steps in between learning updates')

    parser.add_argument('--discount', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--n_step', type=int, default=100,
                       help='Length of rollout')


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
    
    
    # Set up env
    env_cons = lambda: gym.make(args.env)
    
    env = env_cons()
    args.num_actions = env.action_space.n
    args.history_len = 0
    
    # Set agent variables and wrap env based on chosen mode
    mode = args.model
    
    if mode is None:
        shape = env.observation_space.shape
        if len(shape) is 3:
            mode = 'image'
        elif shape[0] is None:
            mode = 'object'
        else:
            mode = 'vanilla'
            
    if mode=='DQN':
        args.model = 'CNN'
        args.history_len = 4
        
        from gym_utils.image_wrappers import ImgGreyScale, ImgResize, ImgCrop
        def wrap_DQN(env):
            env = ImgGreyScale(env)
            env = ImgResize(env, 110, 84)
            return ImgCrop(env, 84, 84)
        env = wrap_DQN(env)
        env_cons = lambda: wrap_DQN(gym.make(args.env))
        
    elif mode=='image':
        args.model = 'CNN'
        args.history_len = 2
        
        from gym_utils.image_wrappers import ImgGreyScale
        def wrap_img(env):
            return ImgGreyScale(env)   
        env = wrap_img(env)
        env_cons = lambda: wrap_img(gym.make(args.env))
        
    elif mode=='object':
        args.model = 'object'
        args.history_len = 0
    elif mode=='vanilla':
        args.model = 'fully connected'
        args.history_len = 0
    args.obs_size = list(env.observation_space.shape)

    # Close env to prevent any weird race conditions   
    env.close()
    
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    tf.set_random_seed(args.seed)
    

    pr = cProfile.Profile()
    with tf.Session(config=config) as sess:
        agent = PPOAgent(sess, env_cons, args)

        writer = tf.summary.FileWriter('./logs', sess.graph)


        for i in range(32):
            agent._add_actor()
        time.sleep(0.1)
        
        
        training_iters = 1000000
        
        ep_reward_last=0
        last_update=time.time()
        for i in trange(training_iters):
            pr.enable()
            agent.Train()
            pr.disable()
            
            new_time=time.time()
            if new_time > last_update + 5:
                last_update = new_time
            if i % 1000 == 0 and i != 0:
                sortby = 'cumulative'
                ps = pstats.Stats(pr).sort_stats(sortby)
                #ps.print_stats()
                ep_rewards = agent.ep_rs
                num_eps = len(ep_rewards[ep_reward_last:])
                if num_eps is not 0:
                    avr_ep_reward = np.mean(ep_rewards[ep_reward_last:])
                    max_ep_reward = np.max(ep_rewards[ep_reward_last:])
                    ep_reward_last = len(ep_rewards)
                tqdm.write("{}, {:>7}/{}it | "\
                    .format(time.strftime("%H:%M:%S"), i, training_iters)
                    +"{:4n} episodes, avr_ep_r: {:4.1f}, max_ep_r: {:4.1f}"\
                    .format(num_eps, avr_ep_reward, max_ep_reward) )
                    
        while agent.actors:
            agent._remove_actor()
