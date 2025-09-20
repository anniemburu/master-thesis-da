from rtdl_revisiting_models import MLP as RTDL_MLP, ResNet, FTTransformer

import inspect


#methods = inspect.getmembers(FTTransformer, predicate=inspect.isfunction)

#for name, _ in methods:
#    print(name)
"""user_methods = [
    name for name, obj in inspect.getmembers(FTTransformer)
    if (inspect.isfunction(obj) or inspect.ismethod(obj) or isinstance(obj, classmethod))
    and obj.__qualname__.startswith(FTTransformer.__name__)
]

print(user_methods)"""

cats = [0, 1, 2, 3, 4]
nums = []

if cats:
    print("cats is not empty")
else:
    print("cats is empty")