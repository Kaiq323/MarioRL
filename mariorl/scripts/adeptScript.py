"""
adeptScript.py

Training mario using the adept interface
"""
from adept.scripts.local import parse_args, main
from mariorl.modules.adeptMario import AdeptMarioEnv

if __name__ == '__main__':
    import adept
    adept.register_env(AdeptMarioEnv)
    main(parse_args())