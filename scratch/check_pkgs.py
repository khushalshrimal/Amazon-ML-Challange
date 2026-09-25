import sys

def check(name):
    try:
        mod = __import__(name)
        ver = getattr(mod, '__version__', 'installed')
        print(f"{name}: INSTALLED ({ver})")
    except ImportError:
        print(f"{name}: NOT INSTALLED")

check('lightgbm')
check('xgboost')
check('sklearn')
