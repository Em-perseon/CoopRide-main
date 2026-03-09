# from algo.il import dqn
# from algo.kl.kl import KL
try:
    from algo.non_nueral.distance import Nearest
    from algo.non_nueral.myopic import Myopic
except Exception:
    Nearest = None
    Myopic = None
