import math
from typing import Optional, List

import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled

from controllers import CONTROLLERS
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, DEL_T, CONTROL_START_IDX, LATACCEL_RANGE, STEER_RANGE, LAT_ACCEL_COST_MULTIPLIER

# tiny env that tracks target and current lat accel
class TinyEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    # t,vEgo,aEgo,roll,targetLateralAcceleration,steerCommand - Obs State
    def __init__(self, model_path: str, segments: List[str], num_segs: int = 100, debug: bool = False, on_pid: bool = False, render_mode: Optional[str] = None):        
        self.model = TinyPhysicsModel(model_path, debug=debug)
        self.on_pid = on_pid
        if self.on_pid:
            self.controller = CONTROLLERS["pid"](0,0,0)
            self.low_action = np.array([-10, -10, -10], dtype=np.float32)              
            self.high_action = np.array([10, 10, 10], dtype=np.float32)
            self.action_space = spaces.Box(low=self.low_action, high=self.high_action, dtype=np.float32)
        else:
            self.controller = CONTROLLERS["train"]()
            self.action_space = spaces.Box(low=STEER_RANGE[0], high=STEER_RANGE[1], dtype=np.float32) 

        self.segments = segments
        self.debug = debug
        self.simulator = None

        self.min_lateral_acceleration_roll = -1.7
        self.max_lateral_acceleration_roll = 1.7

        self.min_lateral_acceleration_target = -7.2
        self.max_lateral_acceleration_target = 7.2

        self.min_lateral_acceleration = LATACCEL_RANGE[0]
        self.max_lateral_acceleration = LATACCEL_RANGE[1]

        self.min_forward_acceleration = -13
        self.max_forward_acceleration = 8

        self.min_forward_velocity = -0.1
        self.max_forward_velocity = 43

        self.low_state = np.array(
            [self.min_lateral_acceleration_roll, self.min_forward_velocity, self.min_forward_acceleration, self.min_lateral_acceleration_target, self.min_lateral_acceleration], dtype=np.float32
        )              
        self.high_state = np.array(
            [self.max_lateral_acceleration_roll, self.max_forward_velocity, self.max_forward_acceleration, self.max_lateral_acceleration_target, self.max_lateral_acceleration], dtype=np.float32
        )
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True       

    def step(self, action: np.ndarray):
        last_lataccel = self.simulator.current_lataccel

        if self.on_pid:
            self.controller.tune(action[0], action[1], action[2])
            self.simulator.step()
        else:
            self.simulator.controller.setSteer(action[0])
            self.simulator.step()

        # Convert a possible numpy bool to a Python bool.
        terminated = False
        # # Penalize going out of bounds. This resulted in a bunch of 'cheating' behavior. ex: The agent would not dare steer too far from the middle/current acceleration
        # terminated = bool(
        #     self.simulator.current_lataccel <= self.min_lateral_acceleration
        #     or self.simulator.current_lataccel >= self.max_lateral_acceleration
        # )
        truncated = self.simulator.done()

        target_lat_accel = self.simulator.target_lataccel_history[-1]
        current_lat_accel = self.simulator.current_lataccel_history[-1]
        last_lat_accel = self.simulator.current_lataccel_history[-2]
       
        lat_accel_cost = (target_lat_accel - current_lat_accel)**2
        jerk_cost = ((current_lat_accel - last_lat_accel)/ DEL_T)**2 
        total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost        
        # cost = math.pow((last_lataccel - self.simulator.current_lataccel)/DEL_T, 2)*100
        # cost = math.pow(target_lataccel-self.simulator.current_lataccel, 2) * 100 * 5
        # # Ideally we dont want this exponential reward with capped punishment, but the agent seems to converge more easily with it.
        # reward = 1/(cost+0.001)
        reward = -total_cost

        if terminated:
            reward = -10000000000000000000
         
        if not truncated:
            state, target = self.simulator.get_state_target(self.simulator.step_idx)
            self.state = np.array([state.roll_lataccel, state.v_ego, state.a_ego, target, self.simulator.current_lataccel], dtype=np.float32)
            info = {}
            if self.on_pid:
                info["pid"] = self.controller.tunings()
        else:
            self.state = self.state
            info = {"costs": self.simulator.compute_cost()}

        if self.render_mode == "human":
            self.render()
        return self.state, reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if self.on_pid:
            self.controller.reset()

        if not self.simulator:
            self.simulator = TinyPhysicsSimulator(self.model, random.choice(self.segments), controller=self.controller, debug=self.debug)
        else:
            self.simulator.data_path = random.choice(self.segments)
            self.simulator.data = self.simulator.get_data(self.simulator.data_path)
            self.simulator.reset()
        while self.simulator.step_idx < CONTROL_START_IDX:
            self.simulator.step()
        state, target = self.simulator.get_state_target(self.simulator.step_idx)
        self.state = np.array([state.roll_lataccel, state.v_ego, state.a_ego,  target, self.simulator.current_lataccel], dtype=np.float32)

        if self.render_mode == "human":
            self.render()
        return self.state, {}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array":
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.max_lateral_acceleration - self.min_lateral_acceleration
        scale = self.screen_width / world_width
        carwidth = 40
        carheight = 20

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        pos = self.state[4]

        gfxdraw.hline(self.surf, 0, self.screen_width, 100, (0, 0, 0))       
        l, r, t, b = -carwidth / 2, carwidth / 2, carheight / 2, -carheight / 2
        cartx = pos * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        car_coords = [(l, b), (l, t), (r, t), (r, b)]
        car_coords = [(c[0] + cartx, c[1] + carty) for c in car_coords]
        gfxdraw.aapolygon(self.surf, car_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, car_coords, (0, 0, 0))

        target = self.state[3]
        flagx = target * scale + self.screen_width / 2.0
        flagy1 = 100
        flagy2 = flagy1 + 50
        gfxdraw.vline(self.surf, int(flagx), int(flagy1), int(flagy2), (0, 0, 0))

        gfxdraw.aapolygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )
        gfxdraw.filled_polygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
