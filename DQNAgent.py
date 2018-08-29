from __future__ import division

import numpy as np
import cv2
import tensorflow as tf

from ops import linear

from Actor import RolloutActor
from gym_utils.replay_memory import ReplayMemory


class DQNAgent(RolloutActor):
    def __init__(self, session, args):
        super(DQNAgent, self).__init__()

        # Set up agent parameters:
        ##################################

        self.name = "DQNAgent"

        # Environment details
        self.obs_size = args.obs_size
        self.n_actions = args.num_actions
        self.viewer = None

        # Reinforcement learning parameters
        self.discount = args.discount
        self.initial_epsilon = args.epsilon
        self.epsilon = self.initial_epsilon
        self.epsilon_final = args.epsilon_final
        self.epsilon_anneal = args.epsilon_anneal
        
        # Double Q
        self.use_double_q = args.double_q
        
        # Contracted bellman updates
        self.use_contracted_bellman = False
        
        # Observe and Look Further:
        # Achieving Consistent Performance on Atari
        self.use_tc_loss = False

        # Training parameters
        self.model_type = args.model
        self.history_len = args.history_len
        self.memory_size = args.replay_memory_size
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.learn_step = args.learn_step
        self.target_update_step = 1000

        # Set up other variables:
        ##################################

        # Running variables
        self.step = 0
        self.started_training = False
        self.ep_rs = []
        self.seed = args.seed
        self.rng = np.random.RandomState(self.seed)
        self.session = session

        # Replay Memory
        self.memory = ReplayMemory(self.memory_size, self.obs_size)

        ##################################


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
            #from networks import relational_object_network
            #self.model = relational_object_network
        
        self._build_graph()
        self.session.run(tf.global_variables_initializer())
        
        ##################################
        

    def _build_graph(self):
    
        # Create placeholders
        ##################################
        
        prepend = [None] if self.history_len == 0 else [ None, self.history_len]
        state_dim = prepend + self.obs_size
        
        self.state = tf.placeholder("float", state_dim)
        self.action = tf.placeholder('int64', [None])
        self.poststate = tf.placeholder("float", state_dim)
        self.reward = tf.placeholder("float", [None])
        self.terminal = tf.placeholder('float', [None])
        
        train_summaries = []

        # Apply model to get output action values:
        ##################################

        # Get action value estimates for normal and target network:
        #   (Apply chosen model and then a final linear layer)
        with tf.variable_scope(self.name + '_pred'):
            emb = self.model(self.state)
            self.pred_qs = linear(tf.nn.relu(emb), self.n_actions)
        with tf.variable_scope(self.name + '_pred', reuse=True):
            emb = self.model(self.poststate)
            self.pred_post_qs = linear(tf.nn.relu(emb), self.n_actions)
        with tf.variable_scope(self.name + '_target', reuse=False):
            emb = self.model(self.poststate)
            self.target_post_qs = linear(tf.nn.relu(emb), self.n_actions)
        #if self.use_tc_loss:
          # Not used since we use the target network value
          #with tf.variable_scope(self.name + '_prev', reuse=False):
          #  emb = model(self.poststate)
          #  self.prev_post_qs = linear(tf.nn.relu(emb), self.n_actions)
          
          
        # Get model weights
        self.pred_weights = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'_pred')
        self.targ_weights = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'_target')
        self.prev_weights = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'_prev')

        ##################################


        # Calculate TD Loss
        ##################################

        # Get relevant action
        action_one_hot = tf.one_hot(self.action, self.n_actions, 1.0, 0.0)
        q_acted = tf.reduce_sum(self.pred_qs * action_one_hot, axis=1)
        self.pred_q = q_acted
        
        for a in range(self.n_actions):
            train_summaries.append(tf.summary.scalar(
                 'action_value/'+str(a), tf.reduce_mean(self.pred_qs[:,a])))
            train_summaries.append(tf.summary.histogram(
                 'action_values/'+str(a), self.pred_qs[:,a]))

        # Get target value
        if self.use_double_q:
            # Predict action with current network
            pred_action = tf.argmax(self.pred_post_qs, axis=1)
            pred_a_oh = tf.one_hot(pred_action, self.n_actions, 1.0, 0.0)
            # Get value of action from target network
            V_t1 = tf.reduce_sum(self.target_post_qs * pred_a_oh, axis=1)
        else:
            V_t1 = tf.reduce_max(self.target_post_qs, axis=1)

        # Zero out target values for terminal states
        V_t1 = V_t1 * (1.0-self.terminal)

        # Bellman equation
        if self.use_contracted_bellman:
            eps = 0.01
            def h(x):
                return tf.sign(x)*(tf.sqrt(tf.abs(x)+1) - 1) + eps*x
            def h_inv(x):
                # This took a long time to work out...
                sgn = tf.sign(x)
                c = (2 * eps*(x + sgn) + sgn) / (2*eps**2)
                d = x * (x+2*sgn) / (eps**2)
                return c - sgn * tf.sqrt( c**2 - d)
            self.target_q = h( self.reward + self.discount * h_inv( V_t1 ) )
        else:
            self.target_q = self.reward + self.discount * V_t1

        self.td_err = tf.stop_gradient(self.target_q) - self.pred_q
        
        td_err_mean = tf.reduce_mean(self.td_err)
        train_summaries.append(tf.summary.scalar('td_err', td_err_mean))

        ##################################


        # Loss Function:
        ##################################

        # Huber loss, from baselines
        total_loss = tf.where(
          tf.abs(self.td_err) < 1.0,
          tf.square(self.td_err) * 0.5,
          (tf.abs(self.td_err) - 0.5))

        if self.use_tc_loss:
            total_loss += tf.square( tf.reduce_max(self.pred_post_qs, axis=1) \
                         - self.target_q )

        # Optimiser
        self.optim = tf.train.AdamOptimizer(
                        self.learning_rate).minimize(total_loss)

        ##################################

        # Update ops
        self.targ_update_op = [tf.assign(t, e) for t, e in
                                   zip(self.targ_weights, self.pred_weights)]
        if self.use_tc_loss:
          self.tc_update_op = [tf.assign(t, e) for t, e in
                                   zip(self.prev_weights, self.pred_weights)]

        # Saver
        self.model_weights = self.pred_weights
        self.saver = tf.train.Saver(self.model_weights)
        
        # Summaries
        self.train_summaries = tf.summary.merge(train_summaries)

        ####################################################################


    def _act(self, state):
        # get Q-values of given state from network
        Qs = self.session.run(self.pred_qs, feed_dict={
                  self.state: [state]})[0]

        action = np.argmax(Qs)
        value = Qs[action]

        # Get action via epsilon-greedy
        if self.rng.rand() < self.epsilon:
          action = self.rng.randint(0, self.n_actions)
          #value = Qs[action] #Paper uses maxQ, uncomment for on-policy updates

        return action, value


    def _save_trajectory(self):
        # Add stored data to replay memory
        for t in range(len(self.trajectory)):
            self.memory.add(self.trajectory.states[t],
                            self.trajectory.actions[t],
                            self.trajectory.rewards[t],
                            self.trajectory.terminals[t] )

    def Train(self):
        return self._update()
        
    def _update(self):
        summaries = None
        if self.training:
            self.step += 1

            # Update Epsilon
            per = min(self.step / self.epsilon_anneal, 1)
            self.epsilon = (1-per)*self.initial_epsilon + per*self.epsilon_final

            if self.memory.count > self.batch_size*2 and \
                   (self.step % self.learn_step) == 0:

                # Get transition sample from memory
                s, a, R, s_, t = self.memory.sample(self.batch_size,
                                                        self.history_len)
                # Run optimization op (backprop)
                summaries = self._train(s, a, R, s_, t)
                

            if self.step % self.target_update_step == 0:
                # Update target_network
                self.session.run(self.targ_update_op)

            if self.use_tc_loss:
                # Update last weights
                self.session.run(self.tc_update_op)
        return summaries
        


    def _train(self, states, actions, rewards, poststates, terminals):
        # train on a given sample

        # set training flag
        self.started_training = True

        # if using lists we need to batch up sample
        if self.obs_size[0] == None:
            states = batch_objects(states)
            poststates = batch_objects(poststates)

        # Train network
        feed_dict = {
          self.state: states,
          self.action: actions,
          self.poststate: poststates,
          self.reward: rewards,
          self.terminal: terminals
        }
        _, summaries = self.session.run([self.optim, self.train_summaries],
                                            feed_dict=feed_dict)
        return summaries


    def Reset(self, obs, train=True):
        super(DQNAgent, self).Reset(obs)

        # Resets agent for start of new episode
        self.training = train

    def GetAction(self):
        action = super(DQNAgent, self).GetAction()
        value = self.trajectory._curr_other
        return action, value

    def Save(self, save_dir):
        # Save model to file
        self.saver.save(self.session, save_dir + '/model.ckpt')

    def Load(self, save_dir):
        # Load model from file
        ckpt = tf.train.get_checkpoint_state(save_dir)
        print("Loading model from {}".format(ckpt.model_checkpoint_path))
        self.saver.restore(self.session, ckpt.model_checkpoint_path)

        # Also need to initialise target network
        self.session.run(self.targ_update_op)


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
    import time
    from tqdm import tqdm, trange
    
    import gym
    import gym_vgdl
    
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
                       
    parser.add_argument('--visualize', action='store_true',
                       help='Set True to enable visualisation')
    parser.add_argument('--record', action='store_true',
                       help='Set True to enable video recording')

    parser.add_argument('--training_iters', type=int, default=1000000,
                       help='Number of training iterations to run for')
    parser.add_argument('--display_step', type=int, default=5000,
                       help='Number of iterations between parameter prints')
    parser.add_argument('--save_step', type=int, default=0,
                       help='Number of steps between model checkpointing,' + \
                       ' leave 0 for no saving')
    parser.add_argument('--log', action='store_true',
                       help='Set to log to tensorboard and save py files')

    parser.add_argument('--learning_rate', type=float, default=0.00025,
                       help='Learning rate for TD updates')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Size of batch for Q-value updates')
    parser.add_argument('--replay_memory_size', type=int, default=100000,
                       help='Size of replay memory')
    parser.add_argument('--learn_step', type=int, default=4,
                       help='Number of steps in-between learning updates')

    parser.add_argument('--double_q', action='store_true',
                       help='Use current network to select action for target')
                       
    parser.add_argument('--discount', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='Initial epsilon')
    parser.add_argument('--epsilon_final', type=float, default=None,
                       help='Final epsilon')
    parser.add_argument('--epsilon_anneal', type=int, default=None,
                       help='Epsilon anneal steps')


                       
    args = parser.parse_args()
    
    
    if args.seed is None:
        args.seed = datetime.datetime.now().microsecond

    if args.epsilon_final == None: args.epsilon_final = args.epsilon
    if args.epsilon_anneal == None: args.epsilon_anneal = args.training_iters


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
        args.model = 'fully connected'
        args.history_len = 0
    args.obs_size = list(env.observation_space.shape)

    # Close env to prevent any weird thread collisions  
    #env.close()
    
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
                  'replay_memory_size',
                  'learn_step',
                  'discount',
                  'epsilon',
                  'epsilon_final',
                  'epsilon_anneal',
                 ]
    
    timestamp = datetime.datetime.now().strftime("%y-%m-%d.%H.%M.%S") + "DQN"
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
        agent = DQNAgent(sess, args)
        
        # Data logging
        log_path = os.path.join('.', 'logs', args.log_dir)
        results_file = os.path.join(log_path, "results.npy")
        results = []
        
        if args.log:
            # Create writer
            writer = tf.summary.FileWriter(log_path, sess.graph)
            ep_rs = tf.placeholder(tf.float32, [None])
            reward_summary = tf.summary.merge(
                [ tf.summary.histogram('rewards', ep_rs), 
                  tf.summary.scalar('mean_rewards', tf.reduce_mean(ep_rs)), ] )
        
            # Copy source files
            import glob
            import shutil
            for file in glob.glob("*.py"):
                shutil.copy(file, log_path)

        
        training_iters = args.training_iters
        display_step = args.display_step
        save_step = args.save_step

        # Create env
        #env = wrapped_env()
        
        if args.record:
           from gym_utils.video_recorder_wrapper import VideoRecorderWrapper
           env = VideoRecorderWrapper(env, log_path)
        
        # Start agent
        state = env.reset()
        agent.Reset(state)
        ep_r = 0
        ep_rewards = []
        ep_reward_last=0
        terminal = False
        
        for step in trange(training_iters):

            # Act, and add
            action, value = agent.GetAction()
            state, reward, terminal, info = env.step(action)
            agent.Update(action, reward, state, terminal)
            
            ep_r += reward

            if args.visualize:
                env.render()
                
            if terminal:
                # Bookeeping
                ep_rewards.append(ep_r)
                ep_r = 0

                # Reset agent and environment
                state = env.reset()
                agent.Reset(state)
                
            summary = agent.Train()
            if args.log and summary is not None:
                writer.add_summary(summary, step)
                

            if step % display_step == 0 and step != 0:
                num_eps = len(ep_rewards[ep_reward_last:])
                
                if num_eps is not 0:
                    rewards = ep_rewards[ep_reward_last:]
                    ep_reward_last = len(ep_rewards)
                    avr_ep_reward = np.mean(rewards)
                    max_ep_reward = np.max(rewards)
                    if args.log:
                        writer.add_summary(sess.run(reward_summary,
                                         feed_dict={ep_rs: rewards}), step)
                                         
                tqdm.write("{}, {:>7}/{}it | "\
                    .format(time.strftime("%H:%M:%S"), step, training_iters)
                    +"{:4n} episodes, avr_ep_r: {:4.1f}, max_ep_r: {:4.1f}"\
                    .format(num_eps, avr_ep_reward, max_ep_reward) )
                    
                if args.log:
                    results.append([ num_eps, avr_ep_reward, max_ep_reward ])
                    np.save(results_file, results)
                    
                    
            if save_step != 0 and step % save_step == 0:
                agent.Save(log_path, step)
