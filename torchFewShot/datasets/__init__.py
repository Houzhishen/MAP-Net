from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .cub_load import cub_load
from .sun_load import sun_load
from .flower_load import flower_load


__imgfewshot_factory = {
        'cub': cub_load,
        'sun': sun_load,
        'flower': flower_load
}


def get_names():
    return list(__imgfewshot_factory.keys()) 


def init_imgfewshot_dataset(name, **kwargs):
    if name not in list(__imgfewshot_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__imgfewshot_factory.keys())))
    return __imgfewshot_factory[name](**kwargs)

