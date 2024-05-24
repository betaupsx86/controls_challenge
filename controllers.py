from simple_pid import PID
from stable_baselines3 import PPO, DDPG
import collections
import numpy as np

class BaseController:
  def update(self, target_lataccel, current_lataccel, state):
    raise NotImplementedError


class OpenController(BaseController):
  def update(self, target_lataccel, current_lataccel, state):
    return target_lataccel


class SimpleController(BaseController):
  def update(self, target_lataccel, current_lataccel, state):
    return (target_lataccel - current_lataccel) * 0.3

class WindowAverageFilter():
  def __init__(self, window_size=5):
    self.window = collections.deque()
    self.window_size = window_size

  def __call__(self, value):
    if len(self.window) >= self.window_size:
      self.window.popleft()
    self.window.append(value)    
    return np.mean(self.window)  
 
class PIDController(BaseController):
  # [0.2943785617935104 , 0.3566981159944914, 0.008014285153525817] with window average filter of 5 
  # [0.05               , 0.7               , -0.004              ] with no filter
  def __init__(self, p=0.05, i=0.7, d=-0.004, filter_output=False, filter=WindowAverageFilter()):
    from tinyphysics import STEER_RANGE, DEL_T
    # TODO: Try proportional_on_measurement to see if it improves overshooting
    self.pid = PID(p, i, d, sample_time=None, proportional_on_measurement=False, differential_on_measurement=False)
    self.pid.output_limits = STEER_RANGE
    self.dt = DEL_T
    self.filter = filter
    self.filter_output = filter_output

  def update(self, target_lataccel, current_lataccel, state):
    self.pid.setpoint = target_lataccel
    steer = self.pid(current_lataccel, dt=self.dt)
    # Reduces overshooting/jerk but only marginally
    if self.filter_output:
      steer = self.filter(steer)
    return steer
  
  def tune(self, p,i,d):
    self.pid.tunings = (p,i,d)

  def tunings(self):
    return self.pid.tunings

  def reset(self,):
    self.pid.reset()
  
class TrainController(BaseController):
  def __init__(self):
    self.steer = 0
  def setSteer(self, steer):
    self.steer = steer
  def update(self, target_lataccel, current_lataccel, state):
    return self.steer
  
class PPO_PIDController(BaseController):
  def __init__(self):
    self.model = PPO.load("./logs/ppo_pid/best_model")
    self.pid_controller = PIDController()
  def update(self, target_lataccel, current_lataccel, state):
    action, _ = self.model.predict(np.array([state.roll_lataccel, state.v_ego, state.a_ego,  target_lataccel, current_lataccel], dtype=np.float32))
    self.pid_controller.tune(action[0], action[1], action[2])    
    return self.pid_controller.update(target_lataccel, current_lataccel, state)
  
class PPOController(BaseController):
  def __init__(self):
    self.model = PPO.load("./logs/ppo/best_model")
  def update(self, target_lataccel, current_lataccel, state):
    action, _ = self.model.predict(np.array([state.roll_lataccel, state.v_ego, state.a_ego,  target_lataccel, current_lataccel], dtype=np.float32))
    return action[0]
  
class DDPGController(BaseController):
  def __init__(self):
    self.model = DDPG.load("./logs/ddpg/best_model")
  def update(self, target_lataccel, current_lataccel, state):
    action, _ = self.model.predict(np.array([state.roll_lataccel, state.v_ego, state.a_ego,  target_lataccel, current_lataccel], dtype=np.float32))
    return action[0]

CONTROLLERS = {
  'open': OpenController,
  'simple': SimpleController,
  'pid': PIDController,
  'train': TrainController,
  'ppo_pid': PPO_PIDController,
  'ppo': PPOController,
  'ddpg': DDPGController,
}
