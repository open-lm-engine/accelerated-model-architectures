# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import inspect
from collections import defaultdict
from typing import Any, Callable

import torch
from tqdm import tqdm

from ..utils import device_synchronize, get_boolean_env_variable
from .cache import get_cutotune_cache
from .config import CutoTuneConfig
from .parameter import CutoTuneParameter


_DEBUG_CUTOTUNE = get_boolean_env_variable("DEBUG_CUTOTUNE", False)
_SEPARATOR = "."
_DEFAULT_WARMUP_ITERATIONS = 5
_BENCHMARK_ITERATIONS = 10


class _CutoTune:
    def __init__(
        self,
        function: Callable,
        configs: list[CutoTuneConfig],
        triggers: set[str],
        warmup_iterations: int,
        benchmark_iterations: int,
        functional_triggers: dict[str, Callable] = {},
        reset_to_zero: dict = {},
    ) -> None:
        assert len(configs) > 0, "no cutotune config is passed"

        self.function = function
        self.configs = configs
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations

        self.signature = inspect.getfullargspec(function)
        self.cutotuneable_parameters = set(self.configs[0].get_key_values().keys())

        self._setup_trigger_map(triggers)

        for config in self.configs:
            assert (
                set(config.get_key_values().keys()) == self.cutotuneable_parameters
            ), "cutotune configs don't match the expected function signature"

        self.functional_triggers = functional_triggers
        self.reset_to_zero = reset_to_zero

        self.filename = inspect.stack()[2].filename.split("cute_kernels")[1][1:]
        self.function_hash = f"{self.filename}->{function.__name__}"

        self.function_cache = {}

    def __call__(self, *args, **kwargs) -> Any:
        override_cutotune_parameters = self._check_all_or_no_args_are_cutotune_parameters(*args, **kwargs)
        lookup_key = self._get_lookup_key(*args, **kwargs)

        best_config = self.function_cache.get(lookup_key, None)

        if best_config is None:
            # bypass cutotune for single config
            if len(self.configs) == 1:
                best_config = self.configs[0]
                best_time = 0
            else:
                best_config, best_time, _ = self._cutotune(*args, **kwargs)

            self.function_cache[lookup_key] = best_config
            get_cutotune_cache().add_config(
                function_hash=self.function_hash, lookup_key=lookup_key, config=best_config
            )

            if _DEBUG_CUTOTUNE and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
                print(
                    f"config {best_config} achieved the best time ({best_time} sec) for {lookup_key} for "
                    f"function {self.function.__name__}"
                )

        output = self.function(
            **self._get_function_arguments(
                config=best_config,
                args=args,
                kwargs=kwargs,
                override_allowed=override_cutotune_parameters,
            )
        )

        return output

    def _check_all_or_no_args_are_cutotune_parameters(self, *args, **kwargs) -> bool:
        num_cutotune_overrideables = 0

        for i in range(len(args)):
            variable_name = self.signature.args[i]

            if isinstance(args[i], CutoTuneParameter) and variable_name in self.cutotuneable_parameters:
                num_cutotune_overrideables += 1

        # accessing kwargs.items() breaks torch.compile in backwards of a custom autograd function
        for variable_name in kwargs:
            if (
                isinstance(kwargs.get(variable_name), CutoTuneParameter)
                and variable_name in self.cutotuneable_parameters
            ):
                num_cutotune_overrideables += 1

        assert num_cutotune_overrideables in [
            0,
            len(self.cutotuneable_parameters),
        ], f"invalid number of CutoTuneParameter arguments, should be either 0 or {len(self.cutotuneable_parameters)}"

        return num_cutotune_overrideables == 0

    def _get_function_arguments(
        self, config: CutoTuneConfig, args: list, kwargs: dict, override_allowed: bool
    ) -> dict:
        # copy the best_config first so we can override with args or kwargs
        result = {variable_name: value for variable_name, value in config.get_key_values().items()}

        for i in range(len(args)):
            variable_name = self.signature.args[i]

            if override_allowed or variable_name not in result:
                result[variable_name] = args[i]

        # accessing kwargs.items() breaks torch.compile in backwards of a custom autograd function
        for variable_name in kwargs:
            if override_allowed or variable_name not in result:
                result[variable_name] = kwargs.get(variable_name)

        return result

    @torch.compiler.set_stance("force_eager")
    @torch.inference_mode()
    def _cutotune(self, *args, **kwargs) -> tuple[CutoTuneConfig, float, list[tuple[CutoTuneConfig, float]]]:
        best_config = None
        best_time = float("inf")

        configs = tqdm(self.configs) if _DEBUG_CUTOTUNE else self.configs
        timed_configs = []

        for config in configs:
            if not config.is_condition_valid(
                **self._get_function_arguments(
                    config=CutoTuneConfig({}), args=args, kwargs=kwargs, override_allowed=False
                )
            ):
                continue

            elapsed_time = self._run_benchmark(
                **self._get_function_arguments(config=config, args=args, kwargs=kwargs, override_allowed=False),
            )

            timed_configs.append((config, elapsed_time))

            if elapsed_time < best_time:
                best_config = config
                best_time = elapsed_time

        assert best_config is not None, "no best_config found, check that at least 1 cutotune config is valid"

        return best_config, best_time, timed_configs

    def _get_lookup_key(self, *args, **kwargs) -> Any:
        lookup_key = []

        def _maybe_add_key(variable_name: str, value) -> None:
            if variable_name not in self.variable_name_trigger_map:
                return

            triggers = self.variable_name_trigger_map[variable_name]

            if isinstance(value, torch.Tensor):
                for func_name, func in triggers:
                    if func is None:
                        assert len(triggers) == 1
                        func = lambda tensor: (tensor.dtype, tensor.size(), tensor.stride())

                    lookup_key.append(f"{variable_name}.{func_name} = {func(value)}")
            else:
                assert len(triggers) == 1
                func_name, func = triggers[0]
                assert (
                    func is None
                ), f"trigger ({variable_name}) is not a tensor and shouldn't have a functional trigger"

                lookup_key.append(f"{variable_name} = {value}")

        for i, value in enumerate(args):
            variable_name = self.signature.args[i]
            _maybe_add_key(variable_name, value)

        for variable_name in kwargs:
            _maybe_add_key(variable_name, kwargs[variable_name])

        # now run the functional triggers
        if len(self.functional_triggers) > 0:
            kwargs = self._get_function_arguments(
                config=CutoTuneConfig({}), args=args, kwargs=kwargs, override_allowed=False
            )

            for variable_name, func in self.functional_triggers.items():
                lookup_key.append(f"{variable_name} = {func(**kwargs)}")

        return str(lookup_key)[1:-1]

    def _run_benchmark(self, **kwargs: dict) -> float:
        device_synchronize()

        for _ in range(self.warmup_iterations):
            self.function(**kwargs)

        # TODO generalize to device Event
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        if len(self.reset_to_zero) > 0:
            elapsed_time = 0

            for _ in range(self.benchmark_iterations):
                start.record()
                self.function(**kwargs)
                end.record()

                device_synchronize()
                elapsed_time += start.elapsed_time(end)

                for variable_name, function in self.reset_to_zero.items():
                    if function is None or function(**kwargs):
                        variable = kwargs[variable_name]
                        assert isinstance(variable, torch.Tensor)

                        variable.zero_()
        else:
            start.record()
            for _ in range(self.benchmark_iterations):
                self.function(**kwargs)
            end.record()

            device_synchronize()
            elapsed_time = start.elapsed_time(end)

        return elapsed_time / self.benchmark_iterations

    def _setup_trigger_map(self, triggers: set[str]) -> None:
        assert isinstance(triggers, set), "triggers should be a set"

        self.variable_name_trigger_map = defaultdict(list)

        for trigger in triggers:
            variable_name, func_name, func = self._parse_trigger(trigger)
            self.variable_name_trigger_map[variable_name].append((func_name, func))

        # filter to remove all triggers if None, this is useful for Tensor based triggers
        for variable_name in self.variable_name_trigger_map:
            if ("info", None) in self.variable_name_trigger_map[variable_name]:
                self.variable_name_trigger_map[variable_name] = [("info", None)]

            assert (
                variable_name in self.signature.args
            ), f"unexpected variable_name ({variable_name}) found in triggers"

        for variable_name in self.cutotuneable_parameters:
            assert (
                variable_name not in self.variable_name_trigger_map
            ), "trigger can't be an instance of CutoTuneParameter"

    def _parse_trigger(self, trigger: str) -> tuple[str, str, Callable]:
        split_trigger = trigger.split(_SEPARATOR)
        variable_name = split_trigger[0]

        if len(split_trigger) == 1:
            func_name = "info"
            func = None
        elif len(split_trigger) == 2:
            func_name = split_trigger[1]

            if func_name == "dtype":
                func = lambda tensor: tensor.dtype
            elif func_name in ["size()", "shape"]:
                func = lambda tensor: tensor.size()
            elif func_name == "stride()":
                func = lambda tensor: tensor.stride()
            elif func_name.startswith("size"):
                dim = int(func_name[5:][:-1])
                func = lambda tensor: tensor.size(dim)
            elif func_name.startswith("shape"):
                dim = int(func_name[6:][:-1])
                func = lambda tensor: tensor.size(dim)
            elif func_name.startswith("stride"):
                dim = int(func_name[7:][:-1])
                func = lambda tensor: tensor.stride(dim)
            else:
                raise ValueError(f"unexpected triggeer found ({trigger})")

        return variable_name, func_name, func

    def __repr__(self):
        return self.function_cache


def cutotune(
    configs: list[CutoTuneConfig],
    triggers: set[str] = set(),
    functional_triggers: dict[str, Callable] = {},
    warmup_iterations: int = _DEFAULT_WARMUP_ITERATIONS,
    benchmark_iterations: int = _BENCHMARK_ITERATIONS,
    reset_to_zero: dict = {},
) -> _CutoTune:
    def inner(function: Callable) -> Callable:
        return _CutoTune(
            function=function,
            configs=configs,
            triggers=triggers,
            warmup_iterations=warmup_iterations,
            benchmark_iterations=benchmark_iterations,
            functional_triggers=functional_triggers,
            reset_to_zero=reset_to_zero,
        )

    return inner
