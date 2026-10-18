"""What every page's routes share: the store, and the HTTP status of each
refusal of the service layer."""

from functools import wraps

from wpt_dataset.errors import DatasetError, NotFound, StaleFile
from wpt_store import StoreError, store_from_settings
from wpt_utils.programs import MissingTool


class Context:
    """The repository the routes serve, and its store (WPT_STORE, from the
    environment or the repository's .env), opened when a route first needs
    it: a source's files are read from the store, as `wpt-dataset unpack`
    reads them."""

    def __init__(self, repo, store=None):
        self.repo, self._store = repo, store

    @property
    def store(self):
        if self._store is None:
            self._store = store_from_settings(self.repo.root / ".env")
        return self._store


def refusals(handler):
    """The handler, answering each refusal with its status and message: 404
    NotFound, 409 StaleFile, 422 any other rule of the service layer, 503 a
    store or a program this machine does not have."""
    @wraps(handler)
    def answer(values, body):
        try:
            return handler(values, body)
        except NotFound as error:
            return 404, {"problems": [str(error)]}
        except StaleFile as error:
            return 409, {"problems": [str(error)]}
        except DatasetError as error:
            return 422, {"problems": [str(error)]}
        except (StoreError, MissingTool) as error:
            return 503, {"problems": [f"{type(error).__name__}: {error}"]}
    return answer


STRING, VERSION = {"type": "string"}, {"type": "string", "description": "the version read; \"\" for a new file"}
PATCH = {"type": "string", "pattern": "^[A-Za-z0-9._-]+$", "description": "the name of the patch that records it"}
SAVED = {"type": "object", "required": ["version"], "properties": {"version": STRING}}
