"""Answer as the assistant: learn to write the assistant's turns of the
conversations tagged `answer-as-assistant` (docs/training.md, "The
answer-as-assistant step"), under AdamW at a constant learning rate of
1e-4, without weight decay. The routers' balancing biases stay as they
are."""

METHOD = "assistant-tokens"
ENVIRONMENT = "conversations"
OPTIMIZER = {"name": "adamw", "settings": {"lr": 1e-4, "weight_decay": 0.0}}
LOCKED = ("balancing",)
SETTINGS = {"tag": "answer-as-assistant"}
