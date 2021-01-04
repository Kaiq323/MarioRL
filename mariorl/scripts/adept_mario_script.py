"""
adept_mario_script.py

Training mario using the adept interface
"""
from adept.scripts.local import parse_args, main
from mariorl.modules.adept_mario_env import AdeptMarioEnv
from mariorl.modules.adept_mario_replay import AdeptMarioReplay

if __name__ == '__main__':
    import adept
    adept.register_env(AdeptMarioEnv)
    adept.register_exp(AdeptMarioReplay)
    main(parse_args())