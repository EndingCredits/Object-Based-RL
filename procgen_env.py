from gym_vgdl import VGDLEnv
import numpy as np

from gym.envs.registration import register
import os

DATA_DIR = os.path.dirname(__file__)


class procGen(VGDLEnv):
    def reset(self):
        self.game.reset()
        level = gen_level()
        #print(level)
        self.game.buildLevel(level)
        self.score_last = self.game.score
        state = self._get_obs()
        return state

register(
    id='vgdl_procgen-v0',
    entry_point=procGen,
    kwargs={
        'game_file': os.path.join( DATA_DIR, 'procgen.txt'),
        'level_file': os.path.join( DATA_DIR, 'procgen_lvl0.txt'),
        'block_size': 5
    },
    timestep_limit=100,
    nondeterministic=True,
)

register(
    id='vgdl_procgen_objects-v0',
    entry_point=procGen,
    kwargs={
        'game_file': os.path.join( DATA_DIR, 'procgen.txt'),
        'level_file': os.path.join( DATA_DIR, 'procgen_lvl0.txt'),
        'obs_type': "objects",
        'notable_sprites': ["avatar", "base"],
        'block_size': 10
    },
    timestep_limit=100,
    nondeterministic=True,
)

def gen_level():
    size_x = 9
    size_y = 9
    avatar_pos = [ np.random.randint(0, size_x), #[5, 5]
                   np.random.randint(0, size_y) ]
    food_pos_list = []
    for i in range(np.random.randint(1, 5)):
        food_x = np.random.randint(0, size_x)
        food_y = np.random.randint(0, size_y)
        food_pos_list.append([food_x, food_y])
    level = ""
    for x in range(size_x):
        for y in range(size_y):
            if [x, y] == avatar_pos:
                char = "A"
            elif [x, y] in food_pos_list:
                char = "X"
            else:
                char = "."
            level += char
        level += '\n'
    return level




class procGenMissile(VGDLEnv):
    def reset(self):
        self.game.reset()
        level = gen_level_missile()
        self.game.buildLevel(level)
        self.score_last = self.game.score
        state = self._get_obs()
        return state

register(
    id='vgdl_procmissile-v0',
    entry_point=procGenMissile,
    kwargs={
        'game_file': os.path.join( DATA_DIR, 'missilecommand.txt'),
        'level_file': os.path.join( DATA_DIR, 'missilecommand_lvl0.txt'),
        'block_size': 5
    },
    timestep_limit=1000,
    nondeterministic=True,
)

register(
    id='vgdl_procmissile_objects-v0',
    entry_point=procGenMissile,
    kwargs={
        'game_file': os.path.join( DATA_DIR, 'missilecommand.txt'),
        'level_file': os.path.join( DATA_DIR, 'missilecommand_lvl0.txt'),
        'obs_type': "objects",
        'notable_sprites': ["avatar", "city", "explosion", "incoming",
                            "incoming_slow", "incoming_fast"],
        'block_size': 10
    },
    timestep_limit=1000,
    nondeterministic=True,
)

def gen_level_missile():
    size_x = 23
    size_y = 11
    avatar_pos = [ np.random.randint(0, size_x), #[5, 5]
                   np.random.randint(3, size_y-3) ]

    enemy_pos_list = np.random.choice(size_x, np.random.randint(1, 5), False)
    level = ""
    for i in range(size_x):
        level += 'm' if i in enemy_pos_list else '.'
    level += '\n'
    for y in range(size_y-2):
        for x in range(size_x):
            if [x, y] == avatar_pos:
                char = "A"
            else:
                char = "."
            level += char
        level += '\n'

    base_pos_list = np.random.choice(size_x, np.random.randint(1, 5), False)
    for i in range(size_x):
        level += 'c' if i in base_pos_list else '.'
    level += '\n'
    return level




import subprocess
class procGenZelda(VGDLEnv):
    def reset(self):
        self.game.reset()
        cmd = ["node", "levelgen.js", "zelda", "afile.txt", "13", "9"]
        level = subprocess.run(cmd, stdout=subprocess.PIPE)\
                    .stdout.decode('utf-8')
        self.game.buildLevel(level)
        self.score_last = self.game.score
        state = self._get_obs()
        return state

register(
    id='vgdl_proczelda-v0',
    entry_point=procGenZelda,
    kwargs={
        'game_file': os.path.join( DATA_DIR, 'zelda.txt'),
        'level_file': os.path.join( DATA_DIR, 'zelda_lvl0.txt'),
        'block_size': 5
    },
    timestep_limit=1000,
    nondeterministic=True,
)

register(
    id='vgdl_proczelda_objects-v0',
    entry_point=procGenZelda,
    kwargs={
        'game_file': os.path.join( DATA_DIR, 'zelda.txt'),
        'level_file': os.path.join( DATA_DIR, 'zelda_lvl0.txt'),
        'obs_type': "objects",
        'notable_sprites': ["goal", "key", "avatar", "nokey", "withkey",
                            "enemy", "monsterQuick", "monsterSlow", "monsterNormal"],
        'block_size': 10
    },
    timestep_limit=1000,
    nondeterministic=True,
)
    

fixed_level = gen_level()
