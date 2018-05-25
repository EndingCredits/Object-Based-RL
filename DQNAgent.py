from __future__ import division

import numpy as np
import scipy
import tensorflow as tf

from ops import linear


class DQNAgent():
    def __init__(self, session, args):

        # Set up agent parameters:
        ##################################

        self.name = "Agent"
        
        # Environment details
        self.obs_size = args.obs_size
        self.n_actions = args.num_actions
        self.viewer = None

        # Reinforcement learning parameters
        self.discount = args.discount
        self.n_steps = args.n_step
        self.initial_epsilon = args.epsilon
        self.epsilon = self.initial_epsilon
        self.epsilon_final = args.epsilon_final
        self.epsilon_anneal = args.epsilon_anneal
        self.use_double_q = False

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
        self.seed = args.seed
        self.rng = np.random.RandomState(self.seed)
        self.session = session

        # Replay Memory
        self.memory = ReplayMemory(self.memory_size, self.obs_size)
        
        ##################################


        # Select appropriate preprocessor:
        ##################################
        if args.preprocessor == 'deepmind':
            self.preproc = deepmind_preprocessor
        elif args.preprocessor == 'grayscale':
            #incorrect spelling in order to not confuse those silly americans
            self.preproc = greyscale_preprocessor
        else:
            #a lambda could be used here, but I think this makes more sense
            self.preproc = default_preprocessor
        ##################################
        
        
        # Select appropriate model and input state shape:
        ##################################
        if self.model_type == 'CNN':
            from networks import deepmind_CNN
            state_dim = [None, self.history_len] + self.obs_size
            model = deepmind_CNN
        elif self.model_type == 'nn':
            from networks import fully_connected_network
            state_dim = [None] + self.obs_size
            model = fully_connected_network
        elif self.model_type == 'object':
            from networks import object_embedding_network2
            state_dim = [None] + self.obs_size
            model = object_embedding_network2
        ##################################
        

        ##### Build Tensorflow graph:
        ####################################################################
        
        # Apply model to get output action values:
        ##################################
        self.state = tf.placeholder("float", state_dim)
        
        # Get action value estimates for normal and target network:
        #   (Apply chosen model and then a final linear layer)
        with tf.variable_scope(self.name + '_pred'):
            emb = model(self.state)
            self.pred_qs = linear(tf.nn.relu(emb), self.n_actions)
        with tf.variable_scope(self.name + '_target', reuse=False):
            emb = model(self.state)
            self.target_pred_qs = linear(tf.nn.relu(emb), self.n_actions)
        
        # Get model weights
        self.pred_weights = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'_pred')
        self.targ_weights = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'_target') 
        
        
        self.action = tf.placeholder('int64', [None])
        action_one_hot = tf.one_hot(self.action, self.n_actions, 1.0, 0.0)
        q_acted = tf.reduce_sum(self.pred_qs * action_one_hot, axis=1)
        self.pred_q = q_acted
        ##################################
        
        # Loss Function:
        ##################################
        self.target_q = tf.placeholder("float", [None])
        self.td_err = self.target_q - self.pred_q
        # Huber loss, from baselines
        total_loss = tf.where(
          tf.abs(self.td_err) < 1.0,
          tf.square(self.td_err) * 0.5,
          (tf.abs(self.td_err) - 0.5))
        ##################################
        
        # Optimiser
        self.optim = tf.train.AdamOptimizer(
                        self.learning_rate).minimize(total_loss)
        
        # Saver
        self.model_weights = self.pred_weights
        self.saver = tf.train.Saver(self.model_weights)

        ####################################################################
        

    def _get_state(self, t=-1):
        # Returns the compiled state from stored observations
        if self.trajectory_t > 0:
            t = t % self.trajectory_t

        if self.history_len == 0:
            # We don't use histories so we can just return current
            state = self.trajectory_observations[t]
        else:
            if self.obs_size[0] == None:
                # using lists, so have to do it slow way
                state = []
                for i in range(self.history_len):
                    state.append(self.trajectory_observations[t-i])
            else:
                # all obs are the same size, can just fill in matrix
                state = np.zeros([self.history_len]+self.obs_size)
                for i in range(self.history_len):
                  if (t-i) >= 0:
                    state[i] = self.trajectory_observations[t-i]
        return state


    def _predict(self, state):
        # get Q-values of given state from network
        qs = self.session.run(self.pred_qs, feed_dict={
                  self.state: [state]})[0]
        return qs
        
        
    def _eval(self, states):
        # get Values (i.e. Q-max) of given states from network
        qs = self.session.run(self.target_pred_qs, feed_dict={
                  self.state: states})
        return np.max(qs, axis=1)


    def _train(self, states, actions, rewards, poststates, terminals):
        # train on a given sample
        
        # set training flag
        self.started_training = True
        
        # if using lists we need to batch up sample
        if self.obs_size[0] == None:
            states, _ = batch_objects(states)
            poststates, _ = batch_objects(poststates)
        
        if self.use_double_q: #N.B: Not used
            # Predict action with current network
            action = np.argmax(self.pred_qs.eval({self.state: states}), axis=1)
            #   neat little trick for getting one-hot:
            action_one_hot = np.eye(self.n_actions)[action]

            # Get value of action from target network
            V_t1 = self.target_pred_qs.eval({self.state: states})
        else:
            V_t1 = self._eval(poststates)
        # Zero out target values for terminal states
        V_t1 = np.multiply(np.ones(np.shape(terminals)) - terminals, V_t1)
        
        # Bellman equation
        Q_targets = self.discount * V_t1 + rewards

        # Train network
        feed_dict = {
          self.state: states,
          self.target_q: Q_targets,
          self.action: actions
        }
        self.session.run(self.optim, feed_dict=feed_dict)
        
        return True


    def Reset(self, obs, train=True):
        # Resets agent for start of new episode
        self.training = train

        #TODO: turn these lists into a proper trajectory object
        self.trajectory_observations = [self.preproc(obs)]
        self.trajectory_values = []
        self.trajectory_actions = []
        self.trajectory_rewards = []
        self.trajectory_t = 0
        return True


    def GetAction(self):
        # Returns action baed on agent's last stored state
        # TODO: Perform calculations on Update, then use saved values to select actions
        
        # Get state of last stored state
        state = self._get_state()

        # Get Q-values
        Qs = self._predict(state)
        action = np.argmax(Qs)
        value = Qs[action]

        # Get action via epsilon-greedy
        if True: #self.training:
          if self.rng.rand() < self.epsilon:
            action = self.rng.randint(0, self.n_actions)
            #value = Qs[action] #Paper uses maxQ, uncomment for on-policy updates

        # Append predicted value to trajectory
        self.trajectory_values.append(value)
        return action, value


    def Update(self, action, reward, obs, terminal=False):
        # Update agent with latest env data
        # N.B: care must be taken to call this exactly once each env step 
        
        # Store latest information
        self.trajectory_actions.append(action)
        self.trajectory_rewards.append(reward)
        self.trajectory_t += 1
        self.trajectory_observations.append(self.preproc(obs))

        # Train agent
        if self.training:
            self.step += 1
            
            # Update Epsilon
            per = min(self.step / self.epsilon_anneal, 1)
            self.epsilon = (1-per)*self.initial_epsilon + per*self.epsilon_final

            if self.memory.count > self.batch_size*2 and (self.step % self.learn_step) == 0:
                # Get transition sample from memory
                s, a, R, s_, t = self.memory.sample(self.batch_size, self.history_len)
                # Run optimization op (backprop)
                self._train(s, a, R, s_, t)

            if self.step % self.target_update_step == 0:
                # Update target_network
                ops = [ self.targ_weights[i].assign(self.pred_weights[i])
                        for i in range(len(self.targ_weights))]
                self.session.run(ops)

            if terminal:
              # Add stored data to replay memory
              for t in xrange(self.trajectory_t):
                self.memory.add(self.trajectory_observations[t],
                                self.trajectory_actions[t],
                                self.trajectory_rewards[t],
                                (t==(self.trajectory_t-1)) )
                    
        return True
        
    def Save(self, save_dir):
        # Save model to file
        self.saver.save(self.session, save_dir + '/model.ckpt')

    def Load(self, save_dir):
        # Load model from file
        ckpt = tf.train.get_checkpoint_state(save_dir)
        print("Loading model from {}".format(ckpt.model_checkpoint_path))
        self.saver.restore(self.session, ckpt.model_checkpoint_path)
        
        # Also need to initialise target network
        ops = [ self.targ_weights[i].assign(self.pred_weights[i])
                for i in range(len(self.targ_weights))]
        self.session.run(ops)


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
        # Create mask...
        masks.append(np.pad(np.array(np.ones((len(l),1)),dtype=np.float32),
            ((0,max_len-len(l)),(0,0)), mode='constant'))
    return out, masks


# Adapted from github.com/devsisters/DQN-tensorflow/
class ReplayMemory:
  def __init__(self, memory_size, obs_size):
    self.memory_size = memory_size
    self.obs_size = obs_size

    if self.obs_size[0] == None:
        self.observations = [None]*self.memory_size
    else:
        self.observations = np.empty([self.memory_size]+self.obs_size, dtype=np.float16)
    self.actions = np.empty(self.memory_size, dtype=np.int16)
    self.rewards = np.empty(self.memory_size, dtype=np.float16)
    self.terminal = np.empty(self.memory_size, dtype=np.bool_)

    self.count = 0
    self.current = 0

  def add(self, obs, action, rewards, terminal):
    self.observations[self.current] = obs
    self.actions[self.current] = action
    self.rewards[self.current] = rewards
    self.terminal[self.current] = terminal

    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.memory_size

  def _get_state(self, index, seq_len):
    # normalize index to expected range, allows negative indexes
    index = index % self.count
    if seq_len == 0:
      state = self.observations[index]
    else:
      if self.obs_size[0] == None:
        state = []
        for i in range(seq_len):
          state.append(self.observations[index-i])
      else:
        state = np.zeros([seq_len]+self.obs_size)
        for i in range(seq_len):
          state[i] = self.observations[index-i]
    return state

  def _uninterrupted(self, start, final):
    # Returns true if given range is not interrupted
    if self.current in range(start+1, final):
        return False
    for i in range(start, final-1):
        if self.terminal[i] == True: return False
    return True

  def sample(self, batch_size, seq_len=0):
    # sample random indexes
    indexes = [] ; prestates = [] ; poststates = []
    watchdog = 0
    while len(indexes) < batch_size:
      while True:
        # find random index
        index = np.random.randint(1, self.count - 1)
        if seq_len is not 0:
          start = index-seq_len
          if not self._uninterrupted(start, index+1):
            continue
        break

      indexes.append(index)
      prestates.append(self._get_state(index, seq_len))
      poststates.append(self._get_state(index+1, seq_len))
      
    indexes = np.array(indexes)
    return prestates, self.actions[indexes], self.rewards[indexes], poststates, self.terminal[indexes+1]



# Preprocessors:
def default_preprocessor(state):
    return state

def greyscale_preprocessor(state):
    #state = cv2.cvtColor(state,cv2.COLOR_BGR2GRAY)/255.
    state = np.dot(state[...,:3], [0.299, 0.587, 0.114])
    return state

def deepmind_preprocessor(state):
    state = greyscale_preprocessor(state)
    #state = np.array(cv2.resize(state, (84, 84)))
    resized_screen = scipy.misc.imresize(state, (110,84))
    state = resized_screen[18:102, :]
    return state

