from .lru import LRU
from .lfu import LFU
from .arc import ARC
from .lecar import LeCaR
from .skull import SKULL

def get_algorithm(alg_name):
    alg_name = alg_name.lower()

    if alg_name == 'lru':
        return LRU
    if alg_name == 'lfu':
        return LFU
    if alg_name == 'arc':
        return ARC
    if alg_name == 'lecar':
        return LeCaR
    if alg_name == 'skull':
        return SKULL
    return None
