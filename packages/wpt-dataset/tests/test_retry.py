import json

from wpt_dataset import retry
from wpt_store import open_store
from wpt_telemetry import Telemetry


def test_a_download_tried_again_inside_a_command_is_in_its_record(tmp_path, capsys):
    # a build's record must show the downloads that failed and were tried
    # again, not only the terminal that printed them
    store = open_store((tmp_path / "store").as_uri())
    attempts = []

    def download():
        attempts.append(1)
        if len(attempts) == 1:
            raise TimeoutError("the server did not answer")
        return b"the file"

    with (Telemetry.start(store, "wpt-dataset-build", [], "d", env_file=tmp_path / ".env", environ={}) as telemetry,
          retry.reporting(telemetry.log)):
        assert retry.call(download, "https://example.org/book.txt", sleep=lambda _: None) == b"the file"

    lines = [json.loads(line) for line in store.read_record(telemetry.folder, "events.jsonl").splitlines()]
    [tried] = [line for line in lines if line["kind"] == "log"]
    assert tried["severity"] == "error" and tried["text"].startswith("https://example.org/book.txt: the server did not")
    assert tried["text"] in capsys.readouterr().err  # still printed
