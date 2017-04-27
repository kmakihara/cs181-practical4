# Imports.
import numpy as np
import numpy.random as npr
import random
from collections import Counter
# from keras.models import Sequential
# from keras.layers import Dense
# from keras import optimizers

from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.gamma = 0.9
        self.eta = 0.2
        self.epsilon = 0.1
        self.jump_Qs = Counter()
        self.swing_Qs = Counter()
        self.gravity = random.sample([-1, -4], 1)[0]

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.gravity = random.sample([-1, -4], 1)[0]

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        if self.last_action and self.last_reward != -10 and self.last_action == 0:
            self.gravity = state["monkey"]["vel"] - self.last_state["monkey"]["vel"]

        tree_dist = state["tree"]["dist"]/50
        tree_bot = state["tree"]["bot"]/50
        monkey_bot = state["monkey"]["bot"]/50

        swing_Q = self.swing_Qs[tree_dist, tree_bot, monkey_bot, self.gravity]
        jump_Q = self.jump_Qs[tree_dist, tree_bot, monkey_bot, self.gravity]

        if self.last_state:
            last_tree_dist = self.last_state["tree"]["dist"]/50
            last_tree_bot = self.last_state["tree"]["bot"]/50
            last_monkey_bot = self.last_state["monkey"]["bot"]/50

            last_swing_Q = self.swing_Qs[last_tree_dist, last_tree_bot, last_monkey_bot, self.gravity]
            last_jump_Q = self.jump_Qs[last_tree_dist, last_tree_bot, last_monkey_bot, self.gravity]

            if self.last_reward == -10:
                target = self.last_reward
            else:
                target = self.last_reward + self.gamma * max(swing_Q, jump_Q)

            print last_swing_Q - target
            new_swing_Q = (1 - self.eta) * last_swing_Q + self.eta * target
            new_jump_Q = (1 - self.eta) * last_jump_Q + self.eta * target

            self.swing_Qs[last_tree_dist, last_tree_bot, last_monkey_bot, self.gravity] = new_swing_Q
            self.jump_Qs[last_tree_dist, last_tree_bot, last_monkey_bot, self.gravity] = new_jump_Q

        action = np.argmax([swing_Q, jump_Q])

        # epsilon greedy
        if npr.rand() < self.epsilon:
            self.last_action = npr.rand() < 0.5
        else:
            self.last_action = action
        self.last_state = state
        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''

    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()

    return


if __name__ == '__main__':

    # Select agent.
    agent = Learner()

    # Empty list to save history.
    hist = []

    # Run games.
    run_games(agent, hist, 100, 1)

    # Save history.
    np.save('hist',np.array(hist))
