from .updown import UpDown
from .xlan import XLAN
from .xtransformer import XTransformer

__factory = {
    'UpDown': UpDown,
    'XLAN': XLAN,
    'XTransformer': XTransformer
}

def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown caption model:", name)
    return __factory[name](*args, **kwargs)