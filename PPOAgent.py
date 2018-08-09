"""
Adapted from NVIDIA GA3C code
"""

from __future__ import division
import numpy as np
import cv2
import os
import tensorflow as tf
from threading import Thread
from multiprocessing import Process, Queue, Value
import time

from Actor import RolloutActor
from ops import linear

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
        if self.model_type == 'CNN':
            from networks import deepmind_CNN
            self.model = deepmind_CNN
        elif self.model_type == 'fully-connected':
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
          with tf.device('cpu:0'):
            self._build_predict_graph()

        # Saver
        self.model_weights = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'_model')
        self.saver = tf.train.Saver(self.model_weights)
            

    def _build_train_graph(self):

        # Build Queue pipeline
        with tf.name_scope('input_pipeline'):
            prepend = [] if self.history_len == 0 else [self.history_len]
            state_dim = prepend + self.obs_size

            tfqueuetype = tf.PaddingFIFOQueue if self.obs_size[0] == None \
                              else tf.FIFOQueue

            self.model_train_queue = tfqueuetype( self.memory_queue_size,
              dtypes = [ tf.float32, tf.int64, tf.float32, tf.float32,
                         tf.float32],
              shapes = [ state_dim, [], [], [], [self.n_actions] ],
              names = [ 'state', 'action', 'advantage', 'return', 'actprob', ],
              name = 'model_train_queue' )
            self.model_train_queue_len = 0

            multi_batch_size = self.batch_size * self.batch_count
            mbs = tf.placeholder_with_default(multi_batch_size, [])
            multi_batch = self.model_train_queue.dequeue_many(mbs) 

            self.state = multi_batch['state']
            self.action = multi_batch['action']
            self.advantage = multi_batch['advantage']
            self.returns = multi_batch['return']
            self.old_probs = multi_batch['actprob']

            # Reuse same placeholders for adding items
            feed_dict = { 'state': self.state,
                          'action': self.action,
                          'advantage': self.advantage,
                          'return': self.returns,
                          'actprob': self.old_probs, }
            
            self.model_enqueue = self.model_train_queue.enqueue_many(
                                            feed_dict, name="queue_data")
        
        
        # Naive implementation of multiple minibatch iterations
        trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                         epsilon=1e-5, name='optimizer')
                                         
        eps = self.clip_ratio
        vf_coef = self.vf_penalty
        ent_coef = self.entrophy_penalty
                                        
        train_ops = []
        train_summaries = []
        idx = tf.range(multi_batch_size)
        
        train_idx = tf.reshape(tf.random_shuffle(idx),
                        [ self.batch_count, -1 ] )

        for i in range(self.train_count):
         with tf.name_scope('epoch_'+str(i)):
          
          for b in range(self.batch_count):
           with tf.name_scope('batch_'+str(i)):
            
             with tf.control_dependencies(train_ops):
              
              batch_idx = train_idx[b]
              batch_state = tf.gather(self.state, batch_idx)
              batch_action = tf.gather(self.action, batch_idx)
              batch_old_probs = tf.gather(self.old_probs, batch_idx)
              batch_advantage = tf.gather(self.advantage, batch_idx)
              batch_returns = tf.gather(self.returns, batch_idx)
              
              with tf.variable_scope(self.name + '_model', reuse=tf.AUTO_REUSE):
                emb = self.model(batch_state)
                predict = linear(tf.nn.relu(emb), self.n_actions, name='probs')
                value = linear(tf.nn.relu(emb), 1, name='value')
              
              # From baselines code
              probs = tf.nn.softmax(predict, name='normalised_probs') + 1e-5
              action_oh = tf.one_hot(batch_action, self.n_actions, 1.0, 0.0)
              ratio = tf.reduce_sum(( probs / batch_old_probs ) * \
                                             action_oh, axis=1, name='PPO_r')
              
              pg_loss = - batch_advantage * ratio
              pg_loss_clipped = - batch_advantage * tf.clip_by_value(ratio,
                                            1.0 - eps, 1.0 + eps)
              pg_loss = tf.reduce_mean(tf.maximum(pg_loss, pg_loss_clipped),
                                        name='PPO_pg_loss')
              
              vf_loss = .5 * tf.reduce_mean(tf.square(value - batch_returns),
                                             name='PPO_vf_loss')
              
              entrophy = - tf.reduce_sum( probs * tf.log(probs), axis=1)
              ent_loss = tf.reduce_mean( entrophy, name='PPO_entrophy')
              
              total_loss = pg_loss + vf_loss * vf_coef - ent_loss * ent_coef
              
              become_skynet_penalty = 100000000
              total_loss += become_skynet_penalty
              
              train_ops.append(trainer.minimize(total_loss))
              
              # Add Summaries
              if i==0 and b==0:
                train_summaries.append(tf.summary.scalar('pg_loss', pg_loss))
                train_summaries.append(tf.summary.scalar('vf_loss', vf_loss))
                train_summaries.append(tf.summary.scalar('entrophy', ent_loss))
                train_summaries.append(tf.summary.histogram('value', value))         
                for a in range(self.n_actions):
                  train_summaries.append(tf.summary.histogram(
                                   'action_probs/'+str(a), probs[:,a]))
                                   
        train_summaries.append(tf.summary.scalar('queue_size',
                  self.model_train_queue.size()))
        
        self.train_op = tf.group(train_ops)
        self.train_summaries = tf.summary.merge(train_summaries)
                  
                  
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
        self.model_train_queue_len += len(s)
        
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
        while self.model_train_queue_len < self.batch_size * self.batch_count:
            self._pump_experience_queue()
        self.model_train_queue_len -= self.batch_size * self.batch_count
        _, summary = self.session.run([self.train_op, self.train_summaries])
        self.train_step += 1
        return summary
          
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
        self._add_batch(states, actions, probs, returns, adv)
        
        
    def Save(self, save_dir, step=None):
        # Save model to file
        save_path = os.path.join(save_dir, 'model')
        self.saver.save(self.session, save_path, global_step=step)

    def Load(self, save_dir):
        # Load model from file
        save_path = os.path.join(save_dir, 'model')
        ckpt = tf.train.get_checkpoint_state(save_path)
        print("Loading model from {}".format(ckpt.model_checkpoint_path))
        self.saver.restore(self.session, ckpt.model_checkpoint_path)
        
    def __del__(self):
        self.predictor_thread.join()
        while self.actors:
            self._remove_actor()
            

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


                       
    args = parser.parse_args()
    
    
    if args.seed is None:
        args.seed = datetime.datetime.now().microsecond


    # Set up env
    env_cons = lambda: gym.make(args.env)
    
    env = env_cons()
    args.num_actions = env.action_space.n
    args.history_len = 0
    
    # Set agent variables and wrap env based on chosen mode
    mode = args.model
    
    # Autodetect
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
        
    elif mode=='objdetect':
        args.model = 'object'
        args.history_len = 0
        from object_detection_wrappers import TestObjectDetectionWrapper
        env = TestObjectDetectionWrapper(env)
        env_cons = lambda: TestObjectDetectionWrapper(gym.make(args.env))
        
    elif mode=='object':
        args.model = 'object'
        args.history_len = 0
    elif mode=='vanilla':
        args.model = 'fully-connected'
        args.history_len = 0
    args.obs_size = list(env.observation_space.shape)

    # Close env to prevent any weird thread collisions  
    env.close()
    
    
    
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
        agent = PPOAgent(sess, env_cons, args)
        
        # Data logging
        log_path = os.path.join('.', 'logs', args.log_dir)
        results_file = os.path.join(log_path, "results.npy")
        results = []
        
        if args.log:
            # Create writer
            writer = tf.summary.FileWriter(log_path, sess.graph)
            ep_rs = tf.placeholder(tf.float32, [None])
            reward_summary = tf.summary.histogram('rewards', ep_rs)
        
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
            if args.log:
                writer.add_summary(summary, step)

            if step % display_step == 0 and step != 0:
                ep_rewards = agent.ep_rs
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
                    .format(time.strftime("%H:%M:%S"), i, training_iters)
                    +"{:4n} episodes, avr_ep_r: {:4.1f}, max_ep_r: {:4.1f}"\
                    .format(num_eps, avr_ep_reward, max_ep_reward) )
                    
                if args.log:
                    results.append([ num_eps, avr_ep_reward, max_ep_reward ])
                    np.save(results_file, results)
                    
            if (step % (display_step*5)) == 0 and args.record:
                agent.actors[-1].should_record.value = 1
                    
            if save_step != 0 and step % save_step == 0:
                agent.Save(log_path, step)

                    
        while agent.actors:
            agent._remove_actor()
