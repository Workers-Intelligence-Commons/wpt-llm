"""Read text: next-token prediction on windows of the dataset's text, the
windows growing stage by stage, under by-part on its multi-step schedule
(docs/training.md, "The read-text step"). A recipe that stops after it
makes a base model."""

METHOD = "next-token"
ENVIRONMENT = "text-windows"
OPTIMIZER = {"name": "by-part"}
LOCKED = ()
