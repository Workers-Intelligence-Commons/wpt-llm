"""Finish tasks: learn to finish the agentic tasks tagged `finish-tasks` by
group relative policy optimization on the model's own scored attempts
(docs/training.md, "The finish-tasks step"), under AdamW at a constant
learning rate of 1e-6, without weight decay. A task with a one-line form
is played in both forms. The routers' weights and balancing biases stay
as they are."""

METHOD = "grpo"
ENVIRONMENT = "agent-tasks"
OPTIMIZER = {"name": "adamw", "settings": {"lr": 1e-6, "weight_decay": 0.0}}
LOCKED = ("routers", "balancing")
SETTINGS = {"tag": "finish-tasks", "terse": True}
