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
        #self.n_steps = args.n_step
        self.initial_epsilon = args.epsilon
        self.epsilon = self.initial_epsilon
        self.epsilon_final = args.epsilon_final
        self.epsilon_anneal = args.epsilon_anneal
        self.use_double_q = False
        self.use_contracted_bellman = False
        self.use_tc_loss = False
        # Observe and Look Further:
        # Achieving Consistent Performance on Atari

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

        
        # Select appropriate model and input state shape:
        ##################################
        if self.model_type == 'CNN':
            from networks import deepmind_CNN
            model = deepmind_CNN
        elif self.model_type == 'fully connected':
            from networks import fully_connected_network
            model = fully_connected_network
        elif self.model_type == 'object':
            from networks import object_embedding_network2
            model = object_embedding_network2
            #from networks import relational_object_network
            #model = relational_object_network
            
        prepend = [None] if self.history_len == 0 else [ None, self.history_len]
        state_dim = prepend + self.obs_size

        ##################################
        

        ##### Build Tensorflow graph:
        ####################################################################
        
        self.state = tf.placeholder("float", state_dim)
        self.action = tf.placeholder('int64', [None])
        self.poststate = tf.placeholder("float", state_dim)
        self.reward = tf.placeholder("float", [None])
        self.terminal = tf.placeholder('float', [None])

        # Apply model to get output action values:
        ##################################


        # Get action value estimates for normal and target network:
        #   (Apply chosen model and then a final linear layer)
        with tf.variable_scope(self.name + '_pred'):
            emb = model(self.state)
            self.pred_qs = linear(tf.nn.relu(emb), self.n_actions)
        with tf.variable_scope(self.name + '_pred', reuse=True):
            emb = model(self.poststate)
            self.pred_post_qs = linear(tf.nn.relu(emb), self.n_actions)
        with tf.variable_scope(self.name + '_target', reuse=False):
            emb = model(self.poststate)
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
        
        # Get target value
        if self.use_double_q:
            # Predict action with current network
            pred_action = tf.argmax(self.pred_post_qs, axis=1)
            # Get value of action from target network
            V_t1 = self.target_post_qs[pred_action]
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

        self.targ_update_op = [tf.assign(t, e) for t, e in
                                   zip(self.targ_weights, self.pred_weights)]
        if self.use_tc_loss:                   
          self.tc_update_op = [tf.assign(t, e) for t, e in
                                   zip(self.prev_weights, self.pred_weights)]
        
        # Saver
        self.model_weights = self.pred_weights
        self.saver = tf.train.Saver(self.model_weights)

        ####################################################################

        
    def _act(self, state):
        # get Q-values of given state from network
        Qs = self.session.run(self.pred_qs, feed_dict={
                  self.state: [state]})[0]
        action = np.argmax(Qs)
        value = Qs[action]

        # Get action via epsilon-greedy
        if True: #self.training:
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
        
        
    def _update(self):
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
                self._train(s, a, R, s_, t)

            if self.step % self.target_update_step == 0:
                # Update target_network
                self.session.run(self.targ_update_op)

            if self.use_tc_loss:
                # Update last weights
                self.session.run(self.tc_update_op)

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
        self.session.run(self.optim, feed_dict=feed_dict)

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
