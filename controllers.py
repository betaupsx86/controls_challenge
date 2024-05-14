from simple_pid import PID
from stable_baselines3 import PPO
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
  
class PIDController(BaseController):
  def __init__(self, p=0.35, i=0.1, d=-0.019):
    from tinyphysics import STEER_RANGE, DEL_T
    self.pid = PID(p, i, d, sample_time=None, proportional_on_measurement=False)
    self.pid.output_limits = STEER_RANGE
    self.dt = DEL_T

  def update(self, target_lataccel, current_lataccel, state):
    self.pid.setpoint = target_lataccel
    steer = self.pid(current_lataccel, dt=self.dt)
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
    self.model = PPO.load("ppo_pid_tiny")
    self.pid_controller = PIDController()
  def update(self, target_lataccel, current_lataccel, state):
    action, _ = self.model.predict(np.array([state.roll_lataccel, state.v_ego, state.a_ego,  target_lataccel, current_lataccel], dtype=np.float32))
    self.pid_controller.tune(action[0], action[1], action[2])    
    return self.pid_controller.update(target_lataccel, current_lataccel, state)
  
class PPOController(BaseController):
  def __init__(self):
    self.model = PPO.load("ppo_tiny")
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
}
