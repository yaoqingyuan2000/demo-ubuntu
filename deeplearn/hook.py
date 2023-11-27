
from registry.registry import *

@HOOKS.register_module(name="Hook")
class Hook():
    def __init__(self, h):
        print("hook----")

    def before_run(self, runner) -> None:
        print("before_run----------------")


    def after_run(self, runner) -> None:
        print("after_run-----------------")