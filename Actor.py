from gym_utils.rollout import Rollout

class RolloutActor(object):
    """
    A basic template class for a generic RL actor with episode history
    """
    
    def __init__(self):
        self.trajectory = Rollout()
        self.history_len = 0
 
    def _get_state(self):
        return self.trajectory.get_state(self.history_len)
    
    def _act(self, state):
        raise NotImplementedError
        
    def _update(self):
        pass
        
    def _save_trajectory(self):
        pass
        
    def Reset(self, obs):
        # Yeild stored trajectory
        self._save_trajectory()
        
        # Make new trajectory
        self.trajectory = Rollout()
        self.trajectory.update_state(obs)

    def GetAction(self):
        # Get state of last stored state and get prediction
        state = self._get_state()
        
        act = self._act(state)
        action = act[0]
        other = act[1:]
        
        # Update episode memory
        self.trajectory.update_action(action)
        self.trajectory.update_other(other)
        return action
        
    def Update(self, action, reward, obs, terminal=False):
        # Update episode memory
        self.trajectory.update_action(action)
        self.trajectory.update_reward(reward, terminal)
        self.trajectory.step()
        self.trajectory.update_state(obs)
        
        # Update (e.g. training)
        self._update()
        
           

            


