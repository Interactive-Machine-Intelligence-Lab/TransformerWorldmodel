import gym
import pygame


def get_keymap_and_action_names(name='null'):

    if name == 'empty':
        return EMPTY_KEYMAP, EMPTY_ACTION_NAMES

    if name == 'episode_replay':
        return EPISODE_REPLAY_KEYMAP, EPISODE_REPLAY_ACTION_NAMES

    if name == 'atari':
        return PUSHBLOCK_KEYMAP, PUSHBLOCK_ACTION_NAMES
    action_names = ['stop', 'forward', 'backwrad', 'turnleft', 'turnright', 'left', 'right']   
    keymap = {}

    for key, value in PUSHBLOCK_KEYMAP.items():
        if PUSHBLOCK_ACTION_NAMES[value] in action_names:
            keymap[key] = action_names.index(PUSHBLOCK_ACTION_NAMES[value])
    return keymap, action_names




PUSHBLOCK_ACTION_NAMES = [
    'stop',
    'forward',
    'backwrad',
    'turnleft',
    'turnright',
    'left',
    'right'
]

PUSHBLOCK_KEYMAP = {
    pygame.K_SPACE: 0,
    pygame.K_w: 1,
    pygame.K_d: 6,
    pygame.K_a: 5,
    pygame.K_s: 2,
    pygame.K_q: 3,
    pygame.K_e: 4,
}

EPISODE_REPLAY_ACTION_NAMES = [
    'stop',
    'forward',
    'backwrad',
    'turnleft',
    'turnright',
    'left',
    'right'
]

EPISODE_REPLAY_KEYMAP = {
    pygame.K_SPACE: 0,
    pygame.K_w: 1,
    pygame.K_d: 6,
    pygame.K_a: 5,
    pygame.K_s: 2,
    pygame.K_q: 3,
    pygame.K_e: 4,
}

EMPTY_ACTION_NAMES = [
    'noop',
]

EMPTY_KEYMAP = {
}