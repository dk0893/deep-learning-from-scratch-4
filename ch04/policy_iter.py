if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from collections import defaultdict
from common.gridworld import GridWorld
from common.image_store import ImageStore
from ch04.policy_eval import policy_eval


def argmax(d):
    """d (dict)"""
    max_value = max(d.values())
    max_key = -1
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key


def greedy_policy(V, env, gamma):
    pi = {}

    for state in env.states():
        action_values = {}

        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values[action] = value

        max_action = argmax(action_values)
        action_probs = {0: 0, 1: 0, 2: 0, 3: 0}
        action_probs[max_action] = 1.0
        pi[state] = action_probs
    return pi


def policy_iter(env, gamma, threshold=0.001, is_render=True):
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0)

    while True:
        V = policy_eval(pi, V, env, gamma, threshold)
        new_pi = greedy_policy(V, env, gamma)

        if is_render:
            ImageStore.st_step( env, V, pi )

        if new_pi == pi:
            break
        pi = new_pi

    return pi


def parse_args():
    
    import argparse
    
    parser = argparse.ArgumentParser( description='policy_iter.py' )
    
    parser.add_argument( '--ope',   default=None,              help='select output operation, [None or im_step or im_state or ani_step or ani_state]' )
    parser.add_argument( '--dpath', default="images",          help='input save image directory path' )
    parser.add_argument( '--fpath', default='policy_iter.gif', help='input save animation path' )
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    ImageStore.init( args.ope, args.dpath, args.fpath )
    
    env = GridWorld()
    gamma = 0.9
    pi = policy_iter(env, gamma)
    
    ImageStore.output( env.renderer.fig )
