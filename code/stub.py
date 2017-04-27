# Imports.
import numpy as np
import numpy.random as npr
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        # self.gamma = 0.9
        # self.eta = 0.2
        # self.epsilon = 0.1
        # self.states = dict()
        #
        # sgd = optimizers.SGD(lr=0.2)
        #
        # self.model_swing = Sequential()
        # self.model_swing.add(Dense(7, input_dim=7, activation='relu', kernel_initializer='random_uniform'))
        # self.model_swing.add(Dense(7, activation='relu', kernel_initializer='random_uniform'))
        # self.model_swing.add(Dense(1))
        # self.model_swing.compile(loss='mse', optimizer=sgd)
        #
        # self.model_jump = Sequential()
        # self.model_jump.add(Dense(7, input_dim=7, activation='relu', kernel_initializer='random_uniform'))
        # self.model_jump.add(Dense(7, activation='relu', kernel_initializer='random_uniform'))
        # self.model_jump.add(Dense(1))
        # self.model_jump.compile(loss='mse', optimizer=sgd)
        #
        # self.replay = []
        # self.first_v = 0
        # self.second_v = 0
        # self.gravity = 0
        self.last_velocity = None

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        # self.first_v = 0
        # self.second_v = 0
        # self.gravity = 2
        self.last_velocity = None

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.
#
#         cur_state = (
#             state['monkey']['top'],
#             state['monkey']['bot'],
#             state['monkey']['vel'],
#             state['tree']['top'],
#             state['tree']['bot'],
#             state['tree']['dist'],
#             self.gravity
#         )
#
#         cur_state_array = np.array([cur_state])
#
# ########
#
#         if self.last_state:
#             last_state = (
#                 self.last_state['monkey']['top'],
#                 self.last_state['monkey']['bot'],
#                 self.last_state['monkey']['vel'],
#                 self.last_state['tree']['top'],
#                 self.last_state['tree']['bot'],
#                 self.last_state['tree']['dist'],
#                 self.gravity
#             )
#
#             last_state_array = np.array([last_state])
#
#             env = (last_state_array, self.last_action, self.last_reward, cur_state_array)
#
#             self.replay.append(env)
#
#             if len(self.replay) >= 10:
#                 replays = npr.choice(range(len(self.replay)), 10, replace=False)
#                 for index in replays:
#                     ss, aa, rr, ss_prime = self.replay[index]
#                     replay_swing_Q = self.model_swing.predict(ss_prime)
#                     replay_jump_Q = self.model_jump.predict(ss_prime)
#
#                     replay_max_Q = max(replay_jump_Q, replay_swing_Q)
#
#                     if rr == -10:
#                         tt = rr
#                     else:
#                         tt = rr + self.gamma * replay_max_Q
#                     if aa == 0:
#                         self.model_swing.train_on_batch(ss, tt)
#                     else:
#                         self.model_jump.train_on_batch(ss, tt)
#
#             else:
#                 update_swing_Q = self.model_swing.predict(cur_state_array)
#                 update_jump_Q = self.model_jump.predict(cur_state_array)
#                 #update_Qs = self.model.predict(cur_state_array)
#                 max_Q = max(update_swing_Q, update_jump_Q)
#                 #max_Q = max(update_Qs[0])
#
#                 if self.last_reward == -10:
#                     target = self.last_reward
#                 else:
#                     target = self.last_reward + self.gamma * max_Q
#
#                 if self.last_action == 0:
#                     self.model_swing.train_on_batch(last_state_array, target)
#                 else:
#                     self.model_jump.train_on_batch(last_state_array, target)
#                 # self.model.train_on_batch(last_state_array, target)
#
# #######
#
#         # Predict:
#         swing_Q = self.model_swing.predict(cur_state_array)
#         jump_Q = self.model_jump.predict(cur_state_array)
#
#         # Qs = self.model.predict(cur_state_array)
#         #
#         # action = np.argmax(Qs[0])
#
#         action = np.argmax([swing_Q, jump_Q])
#
#         self.last_state = state
#
#         # epsilon greedy
#         if npr.rand() < self.epsilon:
#             self.last_action = npr.rand() < 0.5
#         else:
#             self.last_action = action

        # approaching tree
        tree_dist = state["tree"]["dist"]
        act = 0
        if tree_dist > 0 and tree_dist < 100:

            # jump near bottom of tree
            dist_bot_tree = state["monkey"]["bot"] - state["tree"]["bot"]
            if dist_bot_tree < 20:
                act = 1

            #swing near top
            dist_top_tree = state["tree"]["top"] - state["monkey"]["top"]
            if dist_top_tree < 50:
                act = 0

            #jump near bottom of floor
            if state["monkey"]["bot"] < 100:
                act = 1

            #swing near top
            if 400 - state["monkey"]["top"] < 100:
                act = 0

        # not approaching tree
        else:
            #jump near bottom of floor
            if state["monkey"]["bot"] < 100:
                act = 1

            #swing near top
            if 400 - state["monkey"]["top"] < 100:
                act = 0

        # if self.last_action == 0 and act == 0:
        #     gravity = state["monkey"]["vel"] - self.last_velocity
        #
        # self.last_action = act
        # self.last_state = state
        # self.last_velocity = state["monkey"]["vel"]

        return act

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        # last_state = (
        #     self.last_state['monkey']['top']/20,
        #     self.last_state['monkey']['bot']/20,
        #     self.last_state['monkey']['vel']/20,
        #     self.last_state['tree']['top']/20,
        #     self.last_state['tree']['bot']/20,
        #     self.last_state['tree']['dist']/20,
        # )
        #
        # last_state_array = np.array([last_state])
        #
        # last_swing_Q = self.model_swing.predict(last_state_array)
        # last_jump_Q = self.model_jump.predict(last_state_array)
        #
        # max_Q = max(last_swing_Q[0], last_jump_Q[0])
        #
        # action = np.argmax([last_swing_Q[0], last_jump_Q[0]])
        #
        # if last_state not in self.states.keys():
        #     self.states[last_state] = dict()
        #
        # self.states[last_state][self.last_action] = reward + self.gamma * max_Q
        # if reward == -10:
        #     self.reset()

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
    run_games(agent, hist, 100, 10)

    # Save history.
    np.save('hist',np.array(hist))
