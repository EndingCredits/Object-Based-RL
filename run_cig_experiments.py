import tensorflow as tf

import os
import random
import time
from tqdm import tqdm, trange

import gym
import numpy as np
import tensorflow as tf

import DQNAgent

#TODO: Split this into a separate agent initiation of agent and env and training
def run_agent(args):

  # Launch the tensorflow graph
  config = tf.ConfigProto()
  config.gpu_options.allow_growth=True
  with tf.Session(config=config) as sess:

    # Set up training variables
    training_iters = args.training_iters
    display_step = args.display_step
    test_step = args.test_step
    test_count = args.test_count
    tests_done = 0
    test_results = []

    # Stats for display
    ep_rewards = [] ; ep_reward_last = 0
    qs = [] ; q_last = 0
    avr_ep_reward = max_ep_reward = avr_q = 0.0

    # Set precision for printing numpy arrays, useful for debugging
    #np.set_printoptions(threshold='nan', precision=3, suppress=True)

    mode = args.model

    # Create environment
    try:
        import gym_vgdl #This can be found on my github if you want to use it.
    except:
        pass

    env = gym.make(args.env)

    if args.env == 'vgdl_generic-v0':
        #env._obs_type = 'objects'
        from vgdl_game_example import aliens_game, aliens_level
        env.loadGame(aliens_game, aliens_level)

    #N.B: only works with discrete action spaces
    args.num_actions = env.action_space.n

    # Autodetect mode
    if mode is None:
        shape = env.observation_space.shape
        if len(shape) is 3:
            mode = 'DQN'
        elif shape[0] is None:
            mode = 'object'
        else:
            mode = 'vanilla'

    # Set agent variables and wrap env based on chosen mode
    if mode=='DQN':
        args.model = 'CNN'
        args.preprocessor = 'deepmind'
        from gym_utils.image_wrappers import ImgGreyScale, ImgResize, ImgCrop
        env = ImgGreyScale(env)
        env = ImgResize(env, 110, 84)
        env = ImgCrop(env, 84, 84)
        args.obs_size = list(env.observation_space.shape)
        args.history_len = 4
    elif mode=='image':
        args.model = 'CNN'
        args.preprocessor = 'grayscale'
        from gym_utils.image_wrappers import ImgGreyScale, ImgResize, ImgCrop
        env = ImgGreyScale(env)
        args.obs_size = list(env.observation_space.shape)
        args.history_len = 2
    elif mode=='object':
        args.model = 'object'
        args.preprocessor = 'default'
        args.obs_size = list(env.observation_space.shape)
        args.history_len = 0
    elif mode=='vanilla':
        args.model = 'fully connected'
        args.preprocessor = 'default'
        args.obs_size = list(env.observation_space.shape)
        args.history_len = 0

    # Override for object detection from images
    if args.objdetect:
        from object_detection_wrappers import TestObjectDetectionWrapper
        env = TestObjectDetectionWrapper(env)
        args.model = 'object'
        args.preprocessor = 'default'
        args.obs_size = list(env.observation_space.shape)
        args.history_len = 0

    print("Building agent for env shape " + str(args.obs_size))

    # Create agent
    agent = DQNAgent.DQNAgent(sess, args)

    # Initialize all tensorflow variables
    sess.run(tf.global_variables_initializer())


    # Start Agent
    state = env.reset()
    agent.Reset(state)
    rewards = []
    terminal = False

    # Train until we reach max iterations
    for step in trange(training_iters, ncols=80):

        # Act, and add
        action, value = agent.GetAction()
        state, reward, terminal, info = env.step(action)
        agent.Update(action, reward, state, terminal)

        if args.render:
            env.render()

        # Bookeeping
        rewards.append(reward)
        qs.append(value)

        if terminal:
            # Bookeeping
            ep_rewards.append(np.sum(rewards))
            rewards = []

            # Run tests
            if step >= (tests_done)*test_step:
                tests_done += 1
                R_s = run_tests(agent, env, test_count, args.render)

                test_results.append({ 'step': step,
                          'scores': R_s,
                          'average': np.mean(R_s),
                          'max': np.max(R_s) })
                summary = { 'params': vars(args), 'tests': test_results }

                if args.save_file is not None:
                    np.save(args.save_file, summary)

                if args.chk_file is not None:
                    agent.Save(args.chk_file)

            # Reset agent and environment
            state = env.reset()
            agent.Reset(state)


        # Display Statistics
        if (step) % display_step == 0:
            num_eps = len(ep_rewards[ep_reward_last:])
            if num_eps is not 0:
                avr_ep_reward = np.mean(ep_rewards[ep_reward_last:])
                max_ep_reward = np.max(ep_rewards[ep_reward_last:])
                avr_q = np.mean(qs[q_last:]) ; q_last = len(qs)
                ep_reward_last = len(ep_rewards)
            tqdm.write("{}, {:>7}/{}it | "\
                .format(time.strftime("%H:%M:%S"), step, training_iters)
                +"{:3n} episodes, q: {:4.3f}, avr_ep_r: {:4.1f}, "\
                .format(num_eps, avr_q, avr_ep_reward)
                +"max_ep_r: {:4.1f}, epsilon: {:4.3f}"\
                .format(max_ep_reward, agent.epsilon))

    # Continue until end of episode
    step = training_iters
    while not terminal:
        # Act, and add
        action, value = agent.GetAction()
        state, reward, terminal, info = env.step(action)
        agent.Update(action, reward, state, terminal)
        if args.render:
            env.render()
        step += 1

    # Final test
    R_s = run_tests(agent, env, test_count, args.render)
    test_results.append({ 'step': step,
              'scores': R_s,
              'average': np.mean(R_s),
              'max': np.max(R_s) })
    summary = { 'params': vars(args), 'tests': test_results }
    if args.save_file is not None:
        np.save(args.save_file, summary)

    if args.chk_file is not None:
        agent.Save(args.chk_file)


def test_agent(agent, env, render=True):
    try:
        state = env.reset(train=False)
    except:
        state = env.reset()
    agent.Reset(state, train=False)
    R = 0

    terminal = False
    while not terminal:
        action, value = agent.GetAction()
        state, reward, terminal, info = env.step(action)
        agent.Update(action, reward, state, terminal)
        if render:
            env.render()
        R += reward
    return R


def run_tests(agent, env, test_count=50, render=True):
    R_s = []

    # Do test_count tests
    for i in trange(test_count, ncols=50,
      bar_format='Testing: |{bar}| {n_fmt}/{total_fmt}'):
        R = test_agent(agent, env, render)
        R_s.append(R)

    tqdm.write("Tests: {}".format(R_s))
    return R_s

def run_experiments(agent_types, games):
    model_names = { 'image' : 'image', 'objects': 'object', 'features': 'vanilla' }
    suffixes = { 'image' : '', 'objects': '_objects', 'features': '_features' }

    results_dir = 'results'
    try:
        os.stat(results_dir)
    except:
        os.mkdir(results_dir) 


    # We create a class just to get a namespace
    class defaults():
        env_type='gym'
        training_iters=2000000
        display_step=25000
        test_step=50000
        test_count=50
        
        learning_rate=0.00001
        batch_size=32
        replay_memory_size=50000
        learn_step=4
        memory_size=500000
        num_neighbours=50
        alpha=0.25
        delta=0.001

        double_q=False
        
        n_step=100
        discount=0.99
        epsilon=0.5
        epsilon_final=0.1
        epsilon_anneal=500000
        
        chk_file=None
        objdetect=0
        render=1
        
    # Learning rate 0.01, 0.003, 0.001, 0.0003
    # layer sizes 32, 64, 128, 256
    # Num emb layers 2,3,4,5
    # Num out layers 2,3,4,5
    # Initialisation 0.3, 0.1, 0.03, 0.01
    # Replay mem size 100000
    # Normalisation for x and y pos

    lr = [0.00025] #[ 0.01, 0.003, 0.001, 0.0003, 0.0001 ]
    ls = [128] #[ 32, 64, 128, 256 ]
    num_e_l = [4] #[ 2, 3, 4, 5 ]
    num_o_l = [3] #[ 2, 3, 4, 5 ]
    init = [ 0.3, 0.1, 0.03, 0.01 ]
        

    for g in games:
      for a in agent_types:
        args = defaults()
        args.seed = 123
        #N.B: Currently only the environment is seeded, and not the agent.
        
        args.learning_rate = random.choice(lr)
        layer_size = random.choice(ls)
        num_emb_layers = random.choice(num_e_l)
        num_out_layers = random.choice(num_o_l)
        
        # Old code for hyperparameter search
        #name = "lr{}_ls{}_el{}_ol{}".format(args.learning_rate, layer_size,
        #  num_emb_layers, num_out_layers)
        #args.save_file = 'hparamres/' + g + '_' + name
        
        args.emb_layers = [layer_size]*num_emb_layers
        args.out_layers = [layer_size]*num_out_layers
 
        args.save_file = os.path.join(results_dir, g + '_' + a + '_' + str(args.seed))
        args.env = 'vgdl_' + g + suffixes[a] + '-v0'
        args.model = model_names[a]
        
        # Print out arguments
        arg_dict = vars(args)
        col_a_width = 20 ; col_b_width = 32
        print(" ") ; print(" ")
        print(" Running agent with parameters: ")
        print(' ' + '_'*(col_a_width+1+col_b_width) + ' ')
        print('|' + ' '*col_a_width + '|' + ' '*col_b_width  + '|')
        line = "|{!s:>" + str(col_a_width-1) + "} | {!s:<" + str(col_b_width-1) + "}|"
        for i in arg_dict:
            print(line.format(i, arg_dict[i]))
        print('|' + '_'*col_a_width + '|' + '_'*col_b_width  + '|')
        print('')
        
        # Run agent
        run_agent(args)
        tf.reset_default_graph()
        
        
if __name__ == '__main__':
    agent_types = ['image', 'objects', 'features']
    games = ['aliens', 'boulderdash', 'missilecommand', 'survivezombies', 'zelda']
    
    run_experiments(agent_types, games)

