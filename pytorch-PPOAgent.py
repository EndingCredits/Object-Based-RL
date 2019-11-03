"""
Adapted from NVIDIA GA3C code
"""

from __future__ import division
import numpy as np
import os

from threading import Thread
from multiprocessing import Process, Queue, Value
from multiprocessing.pool import ThreadPool
import time

from Actor import RolloutActor

import tensorflow as tf

import torch
import torch.nn as nn
import torch.nn.functional as F

from json2vec import JSONTreeLSTM

class MLP(nn.Module):
    def __init__(self, in_shape, num_actions):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(in_shape[-1], 128)
        self.fc2 = nn.Linear(128, 128)

        self.predict = nn.Linear(128, num_actions)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.predict(x), self.value(x)

class JSON_NN(nn.Module):
    def __init__(self, in_shape, num_actions):
        super(JSON_NN, self).__init__()

        self.tree = JSONTreeLSTM(mem_dim=64)
        self.predict = nn.Linear(128, num_actions)
        self.value = nn.Linear(128, 1)


    def forward(self, x):
        
        #with ThreadPool(8) as p:
        #    elems = p.map(self.tree, x)
        elems = [self.tree(z) for z in x]
        x = torch.cat( elems, dim=0 )
        
        return self.predict(x), self.value(x)


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
        self.lam = args.lam
        self.n_steps = args.n_step
        
        # Double Q
        # Addressing Function Approximation Error in Actor-Critic Methods
        #     - Fujimoto et al.
        #self.use_double_q = args.double_critic

        # Training parameters
        self.train_count = args.num_epoch
        self.batch_count = args.num_batches
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate

        self.clip_ratio = args.clip_ratio
        self.vf_penalty = args.vf_penalty
        self.entrophy_penalty = args.entrophy_penalty

        self.model_type = args.model
        self.history_len = args.history_len

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
        self.num_test_actors = 0
        self.test_actors = []

        self.env = env_constructor

        # Thread queues
        self.trajectories_queue_size = 20
        self.memory_queue_size = self.batch_size*self.batch_count*20
        self.prediction_queue = Queue()
        self.training_queue = Queue(self.trajectories_queue_size)
        self.testing_queue = Queue(self.trajectories_queue_size)
        self.experience_memory = []

        # Predictor
        self.predictor_thread = PredictorThread( self )

        # Set up model:
        ##################################
        self.model = JSON_NN(self.obs_size, self.n_actions)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
          eps=1e-5)
        
        # Start up threads:
        ##################################
        self.predictor_thread.start()

        
    def _get_action(self, state):
        if self.obs_size[0] == None:
            state = batch_objects(state)

        with torch.no_grad():
            if not self.obs_size == [1,]:
                state = torch.Tensor( state )
            preds, value = self.model(state)

            #print(preds.shape, value.shape)

            probs = F.softmax(preds, dim=1).numpy()

            num_act = np.shape(probs)[1]
            try:
                act = [ np.random.choice(range(num_act), p=prob ) for prob in probs ]
            except:
                print(probs)

            return act, probs, value.numpy()[..., 0]

        
    def _add_actor(self, rendering=0, record=None):
        actor = ActorProcess(self.num_actors,
                             self.prediction_queue,
                             self.env,
                             self.training_queue,
                             self.history_len,
                             render=rendering,
                             record=record )
        self.actors.append(actor)
        self.actors[-1].start()
        self.num_actors += 1
    
    def _remove_actor(self):
      if self.num_actors > 0:
        self.num_actors -= 1
        self.actors[-1].exit_flag.value = True
        self.actors[-1].join()
        self.actors.pop()
                 

    def Train(self):

        # Get training batches from training queue
        while len(self.experience_memory) < self.batch_size * self.batch_count:
            self._pump_experience_queue()

        batches = []
        for b in range(self.batch_count):
            batch = []
            for j in range(self.batch_size):
                ind = np.random.randint(len(self.experience_memory))
                batch.append(self.experience_memory.pop(ind))
            batches.append(batch)

        # rename of variables for convenience                         
        eps = self.clip_ratio
        vf_coef = self.vf_penalty
        ent_coef = self.entrophy_penalty  

        for i in range(self.train_count):
          for batch in batches:

            batch_state = [ e[0] for e in batch ]
            if self.obs_size[0] == None:
                batch_state = batch_objects(batch_state)
            if not self.obs_size == [1,] :
                batch_state = torch.Tensor( batch_state )
            batch_action = torch.Tensor([ e[1] for e in batch ])
            batch_old_probs = torch.Tensor([ e[2] for e in batch ])
            batch_advantage = torch.Tensor([ e[3] for e in batch ])
            batch_returns = torch.Tensor([ e[4] for e in batch ])
        
            # Forward pass
            predict, value = self.model(batch_state)
            
            # Policy gradient loss, with PPO constraint
            probs = F.softmax(predict, dim=1) + 1e-5
            action_oh = F.one_hot(batch_action.long(), self.n_actions).float()
            ratio = torch.sum( ( probs / batch_old_probs ) *  action_oh, dim=1)

            pg_loss = - batch_advantage * ratio
            pg_loss_clipped = - batch_advantage * torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
            pg_loss = torch.max(pg_loss, pg_loss_clipped).mean()
            
            # Critic (value function) loss
            vf_loss = .5 * (value - batch_returns).pow(2).mean()
            
            # Entrophy loss
            entrophy = - torch.sum( probs * torch.log(probs), dim=1)
            ent_loss = entrophy.mean()
            
            # Total loss
            total_loss = pg_loss + vf_loss * vf_coef - ent_loss * ent_coef

            become_skynet_penalty = 100000000
            #total_loss += become_skynet_penalty

            # Reward prediction loss
            #rewards = tf.pad(
            #         batch_returns, [[1,0]]) - tf.pad(batch_returns, [[0,1]])
            #rewards = rewards[:-2]
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()


        self.train_step += 1
        return None
          


    def _pump_experience_queue(self):  
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
        returns = np.array(traj.gamma_return(discount=self.discount,
                                             gamma=self.lam, value_index=1))
        values = np.array([ o[1] for o in traj.others ])
        probs = np.array([ o[0] for o in traj.others ])
        adv = returns - values
        
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)

        for j in range(len(states)):
          self.experience_memory.append(
            (states[j], actions[j], probs[j], adv[j], returns[j]) )


 


    def Save(self, save_dir, step=None):
        # Save model to file
        raise NotImplementedError

    def Load(self, save_dir):
        # Load model from file
        raise NotImplementedError
        
    def __del__(self):
        while self.actors:
            self._remove_actor()
        time.sleep(100)
        self.predictor_thread.join()
            


class ActorProcess(Process, RolloutActor):
    def __init__(self,
                 id,
                 prediction_queue,
                 env_constructor,
                 training_queue,
                 state_seq_len=1,
                 render=0,
                 record=None ):
        Process.__init__(self)
        RolloutActor.__init__(self)
        self.daemon = True
        
        self.id = id
        self.prediction_output = prediction_queue
        self.prediction_input = Queue(maxsize=1)
        self.training_queue = training_queue
        self.exit_flag = Value('i', 0)
        self.should_record = Value('i', 0)
        
        self.history_len = state_seq_len
        
        self.render_flag = Value('i', render)
        
        self.env_constructor = env_constructor
        if record is not None:
            from gym_utils.video_recorder_wrapper import VideoRecorderWrapper
            self.env_constructor = lambda: VideoRecorderWrapper(
                env_constructor(), record, should_capture=self._should_record)
            

    def _act(self, state):
        self.prediction_output.put((self.id, state))
        act, probs, values = self.prediction_input.get(True, 60)
        return act, probs, values

    def _save_trajectory(self):
        self.training_queue.put(self.trajectory)
        
    def _should_record(self, t=0):
        if self.should_record.value:
            self.should_record.value = 0
            return True
        return False 

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
                if self.render_flag.value: time.sleep(1.0)
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
    import datetime
    from tqdm import tqdm, trange
    
    import gym
    import gym_vgdl

    import procgen_env
    
    # Uncomment to throw numpy warnings
    #import warnings
    #warnings.filterwarnings("error", category=RuntimeWarning)
    #np.seterr(all='raise')
    
    # Enable headless use of 'headed' environments
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

    parser.add_argument('--seed', type=int, default=None,
                       help='Seed to initialise the agent with')
                       
    parser.add_argument('--num_actors', type=int, default=32,
                       help='Number of actor processes')
    parser.add_argument('--visualize', action='store_true',
                       help='Set True to enable visualisation')
    parser.add_argument('--record', action='store_true',
                       help='Set True to enable video recording')

    parser.add_argument('--training_iters', type=int, default=25000,
                       help='Number of training iterations to run for')
    parser.add_argument('--display_step', type=int, default=100,
                       help='Number of iterations between parameter prints')
    parser.add_argument('--save_step', type=int, default=0,
                       help='Number of steps between model checkpointing,' + \
                       ' leave 0 for no saving')
    parser.add_argument('--log', action='store_true',
                       help='Set to log to tensorboard and save py files')

    parser.add_argument('--learning_rate', type=float, default=0.00025,
                       help='Learning rate for TD updates')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Size of batch for Q-value updates')
    parser.add_argument('--num_epoch', type=int, default=4,
                       help='Number of training steps for each training update')
    parser.add_argument('--num_batches', type=int, default=4,
                       help='Number of training steps for each training update')
                       
    parser.add_argument('--clip_ratio', type=float, default=0.05,
                       help='Maximum policy deviation (epsilon) allowed')
    parser.add_argument('--vf_penalty', type=float, default=1.0,
                       help='Weight of the value-function penalty term in ' +\
                       'the training loss')
    parser.add_argument('--entrophy_penalty', type=float, default=0.01,
                       help='Weight of the entrophy penalty term in ' +\
                       'the training loss')

    parser.add_argument('--discount', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--lam', type=float, default=0.95,
                       help='Lambda factor for weighting returns')
    parser.add_argument('--n_step', type=int, default=100,
                       help='Length of rollout (not used)')
                       
    #parser.add_argument('--double_critic', action='store_true',
    #                   help='Set to use double critic')


                       
    args = parser.parse_args()
    
    
    if args.seed is None:
        args.seed = datetime.datetime.now().microsecond


    # Set up env
    env_cons = lambda: gym.make(args.env)
    wrappers = []

    env = env_cons()
    args.num_actions = env.action_space.n
    args.history_len = 0
    
    # Set agent variables and wrap env based on chosen mode
    mode = args.model


    # Prewrapping for certain envs
    from gym.envs.atari.atari_env import AtariEnv
    if type(env.unwrapped) is AtariEnv:
        from gym_utils.third_party.atari_wrappers import MaxAndSkipEnv
        from gym_utils.third_party.atari_wrappers import ClipRewardEnv
        from gym_utils.third_party.atari_wrappers import EpisodicLifeEnv
        def wrap_atari(env):
            env = MaxAndSkipEnv(env)
            env = ClipRewardEnv(env)
            env = EpisodicLifeEnv(env)
            return env
        env = wrap_atari(env)
        wrappers.append(wrap_atari)
        
    # Autodetect observation type
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
        wrappers.append(wrap_DQN)

    elif mode=='image':
        args.model = 'CNN'
        args.history_len = 2
        
        from gym_utils.image_wrappers import ImgGreyScale
        def wrap_img(env):
            return ImgGreyScale(env)   
        env = wrap_img(env)
        wrappers.append(wrap_img)
        
    elif mode=='objdetect':
        args.model = 'object'
        args.history_len = 0
        from object_detection_wrappers import TestObjectDetectionWrapper
        env = TestObjectDetectionWrapper(env)
        wrappers.append(TestObjectDetectionWrapper)
        
    elif mode=='object':
        args.model = 'object'
        args.history_len = 0
    elif mode=='vanilla':
        args.model = 'fully-connected'
        args.history_len = 0
    args.obs_size = list(env.observation_space.shape)

    # Close env to prevent any weird thread collisions  
    env.close()
    
    def wrap_env(env, wrappers):
        for wrapper in wrappers:
            env = wrapper(env)
        return env
        
    wrapped_env = lambda: wrap_env(env_cons(), wrappers)
       
    
    arg_dict = vars(args)
    train_args = [
                  'env',
                  'model',
                  'history_len',
                  'learning_rate',
                  'batch_size',
                  'num_epoch',
                  'num_batches',
                  'clip_ratio',
                  'vf_penalty',
                  'entrophy_penalty',
                  'discount',
                  'lam',
                  #'double_critic',
                 ]
    
    timestamp = datetime.datetime.now().strftime("%y-%m-%d.%H.%M.%S")
    args.log_dir = '_'.join([timestamp]+[str(arg_dict[arg]) for arg in train_args])
    
    col_a_width = 20 ; col_b_width = 16
    print(' ' + '_'*(col_a_width+1+col_b_width) + ' ')
    print('|' + ' '*col_a_width + '|' + ' '*col_b_width  + '|')
    line = "|{!s:>" + str(col_a_width-1) + "} | {!s:<" + str(col_b_width-1) + "}|"
    for i in arg_dict:
        print(line.format(i, arg_dict[i]))
    print('|' + '_'*col_a_width + '|' + '_'*col_b_width  + '|')
    print('')
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    tf.set_random_seed(args.seed)
    with tf.Session(config=config) as sess:
        agent = PPOAgent(sess, wrapped_env, args)
        
        # Data logging
        log_path = os.path.join('.', 'logs', args.log_dir)
        results_file = os.path.join(log_path, "results.npy")
        results = []
        
        if args.log:
            # Create writer
            ep_rs = tf.placeholder(tf.float32, [None])
            reward_summary = tf.summary.merge(
                [ tf.summary.histogram('rewards', ep_rs), 
                  tf.summary.scalar('mean_rewards', tf.reduce_mean(ep_rs)), ] )
            writer = tf.summary.FileWriter(log_path, sess.graph)
        
            # Copy source files
            import glob
            import shutil
            for file in glob.glob("*.py"):
                shutil.copy(file, log_path)

        # Create actors
        for step in range(args.num_actors-1):
            agent._add_actor()
        record = log_path if args.record else None
        agent._add_actor(rendering=args.visualize, record=record)
        time.sleep(0.1)
        
        training_iters = args.training_iters
        display_step = args.display_step
        save_step = args.save_step
        ep_reward_last=0
        
        for step in trange(training_iters):
            summary = agent.Train()
            #if args.log:
            #    writer.add_summary(summary, step)

            if step % display_step == 0 and step != 0:
                ep_rewards = agent.ep_rs
                num_eps = len(ep_rewards[ep_reward_last:])
                
                if num_eps is not 0:
                    rewards = ep_rewards[ep_reward_last:]
                    ep_reward_last = len(ep_rewards)
                    avr_ep_reward = np.mean(rewards)
                    max_ep_reward = np.max(rewards)
                    std = np.std(rewards)
                    if args.log:
                        writer.add_summary(sess.run(reward_summary,
                                         feed_dict={ep_rs: rewards}), step)
                        results.append([ num_eps, avr_ep_reward,
                                         std, max_ep_reward ])
                        np.save(results_file, results)
                                         
                tqdm.write("{}, {:>7}/{}it | "\
                    .format(time.strftime("%H:%M:%S"), step, training_iters)
                    +"{:4n} episodes, avr_ep_r: {:4.1f}, max_ep_r: {:4.1f}"\
                    .format(num_eps, avr_ep_reward, max_ep_reward) )
                    
                    
            if (step % (display_step*5)) == 0 and args.record:
                agent.actors[-1].should_record.value = 1
                    
            if save_step != 0 and step % save_step == 0:
                agent.Save(log_path, step)