"""Finish tasks with less writing: reach the same result as the model the
step starts from, writing fewer tokens, on the tasks finish-tasks learns
(tagged `finish-tasks`; docs/training.md, "The
finish-tasks-with-less-writing step"). It comes after finish-tasks in a
recipe. The model first makes 8 reference attempts at each task; then
finish-tasks's learning, each task in its full form only, a passing
attempt's reward growing as it gets shorter than the reference's."""

METHOD = "grpo"
ENVIRONMENT = "agent-tasks"
OPTIMIZER = {"name": "by-part", "settings": {"lr": 1e-6, "weight_decay": 0.0, "schedule": "constant", "patience": 10}}
LOCKED = ("routers", "balancing")
SETTINGS = {"tag": "finish-tasks", "reward": "shorter", "reference": 8}
