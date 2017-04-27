# Imports.
import numpy as np
import numpy.random as npr

from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.last_velocity = None
        self.gravity = None

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.last_velocity = None
        self.gravity = None

    def four_gravity_action(self, state):
        act = 0
        # approaching tree
        tree_dist = state["tree"]["dist"]
        if tree_dist > 0 & tree_dist < 100:

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
        return act

    def one_gravity_action(self, state):
        act = 0
        # approaching tree
        tree_dist = state["tree"]["dist"]
        if tree_dist > 0 & tree_dist < 300:

            # jump near bottom of tree
            dist_bot_tree = state["monkey"]["bot"] - state["tree"]["bot"]
            if dist_bot_tree < 20:
                act = 1

            #swing near top
            dist_top_tree = state["tree"]["top"] - state["monkey"]["top"]
            if dist_top_tree < 100:
                act = 0

            #jump near bottom of floor
            if state["monkey"]["bot"] < 10:
                act = 1

            #swing near top
            if 400 - state["monkey"]["top"] < 200:
                act = 0

        # not approaching tree
        else:
            #jump near bottom of floor
            if state["monkey"]["bot"] < 10:
                act = 1

            #swing near top
            if 400 - state["monkey"]["top"] < 200:
                act = 0
        return act

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        gravity = None

        tree_dist = state["tree"]["dist"]

        act = 0

        if self.gravity == None or self.gravity == -4:
            # # approaching tree
            # if tree_dist > 0 & tree_dist < 100:

            #     # jump near bottom of tree
            #     dist_bot_tree = state["monkey"]["bot"] - state["tree"]["bot"]
            #     if dist_bot_tree < 20:
            #         act = 1

            #     #swing near top
            #     dist_top_tree = state["tree"]["top"] - state["monkey"]["top"]
            #     if dist_top_tree < 50:
            #         act = 0

            #     #jump near bottom of floor
            #     if state["monkey"]["bot"] < 100:
            #         act = 1

            #     #swing near top
            #     if 400 - state["monkey"]["top"] < 100:
            #         act = 0

            # # not approaching tree
            # else:
            #     #jump near bottom of floor
            #     if state["monkey"]["bot"] < 100:
            #         act = 1

            #     #swing near top
            #     if 400 - state["monkey"]["top"] < 100:
            #         act = 0
            #act = self.four_gravity_action(state)
            act = 0
        else:
            act = self.one_gravity_action(state)
            print 'one!'

        if self.last_action == 0 & act == 0:
            self.gravity = state["monkey"]["vel"] - self.last_velocity


        #new_action = npr.rand() < 0.1
        new_state  = state

        self.last_action = act
        self.last_state  = new_state
        self.last_velocity = state["monkey"]["vel"]

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
	run_games(agent, hist, 20, 10)

	# Save history.
	np.save('hist',np.array(hist))
