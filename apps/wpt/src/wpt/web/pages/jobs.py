"""The jobs page's routes: start a `wpt` subcommand as a job and follow it
(jobs.py). One route starts each job of the command line, with the
arguments the request gives, as `wpt SUBCOMMAND ARGS...` would take them
(`wpt SUBCOMMAND --help` lists them). Left out are the `wpt dataset`
commands that read or write a person's text (`read`, `write`,
`conversation`, `save-conversation`, `task`, `save-task`, `save-rating`,
`save-probe`, which keep only their editing routes), the options of
`new-source` that do (`--file`, `--license-text`), `wpt web` itself, and
`wpt chat` but for `--about`, since a job reads no standard input."""

from ..jobs import Jobs
from ..server import Route
from .common import STRING, refusals

# the options of `wpt dataset new-source` that carry a person's text, and every prefix argparse takes for them
PERSONS_TEXT = "^--(fi|fil|file|license-|license-t|license-te|license-tex|license-text)(=|$)"

# route name -> (subcommand words, what it does, extra rule on each argument)
JOBS = {
    "dataset/check": (("dataset", "check"), "every rule that needs no download; with --fetch, also download "
                      "sources and check them", None),
    "dataset/check-commits": (("dataset", "check-commits"), "DCO sign-off and signed decisions of a range of "
                              "commits", None),
    "dataset/source-id": (("dataset", "source-id"), "the id a new source gets", None),
    "dataset/manifest": (("dataset", "manifest"), "download a source (or find it in the store) and print its "
                         "upstream tree's manifest", None),
    "dataset/cut": (("dataset", "cut"), "erase bytes of an upstream file from every build, as the source's next "
                    "cut", None),
    "dataset/unpack": (("dataset", "unpack"), "write a source's final files into a new folder outside the "
                       "repository", None),
    "dataset/patch": (("dataset", "patch"), "record a folder's differences from a source's final files as its "
                      "next patch", None),
    "dataset/markdown-check": (("dataset", "markdown-check"), "check the Markdown made from text, xhtml and html "
                               "files", None),
    "dataset/download": (("dataset", "download"), "fetch sources into the store", None),
    "dataset/exclude-unusable": (("dataset", "exclude-unusable"), "write every unusable file of a source as an "
                                 "exact-path exclude", None),
    "dataset/build-tokenizer": (("dataset", "build-tokenizer"), "train (or find) the vocabulary a dataset "
                                "recipe names", None),
    "dataset/tokenizer-check": (("dataset", "tokenizer-check"), "encode and decode every admitted document with a "
                                "vocabulary", None),
    "dataset/build": (("dataset", "build"), "build (or find) the data of a model recipe; prints the dataset id",
                      None),
    "dataset/contamination": (("dataset", "contamination"), "the runs of words a dataset's held-out documents share "
                              "with its training documents, saved in the store", None),
    "dataset/migrate": (("dataset", "migrate"), "rewrite data files to the current shapes", None),
    "dataset/kinds": (("dataset", "kinds"), "the kinds of data files, and who writes each", None),
    "dataset/list": (("dataset", "list"), "the names of the files of a kind", None),
    "dataset/show": (("dataset", "show"), "print a data file", None),
    "dataset/series": (("dataset", "series"), "the sizes a series file derives", None),
    "dataset/save": (("dataset", "save"), "write a data file from a file of this machine", None),
    "dataset/status": (("dataset", "status"), "git's status of every changed file under data/", None),
    "dataset/new-source": (("dataset", "new-source"), "a new empty source under review, without the options that "
                           "carry a person's text (--file, --license-text)", {"not": {"pattern": PERSONS_TEXT}}),
    "dataset/files": (("dataset", "files"), "a source's final files", None),
    "dataset/remove": (("dataset", "remove"), "remove a file of an empty source, as the next patch", None),
    "dataset/tags": (("dataset", "tags"), "give exactly one path its own tags", None),
    "dataset/add-license-text": (("dataset", "add-license-text"), "add the license text found in a source as a new "
                                 "license text under review", None),
    "dataset/conversations": (("dataset", "conversations"), "the conversations of a chat file: messages, roles, "
                              "drafts", None),
    "dataset/tasks": (("dataset", "tasks"), "the task folders of a source: image, limits, files", None),
    "dataset/ratings": (("dataset", "ratings"), "the ratings of a rating file: the attempt each rates, who, when",
                        None),
    "dataset/probes": (("dataset", "probes"), "the probes of a probe file: how many patterns each holds", None),
    "store/list": (("store", "list"), "every session and every source the store database knows", None),
    "store/runs": (("store", "runs"), "every run recorded in the store", None),
    "store/run": (("store", "run"), "one run: its stage, counts, last events and errors; --follow prints its lines "
                  "until it ends", None),
    "store/mark": (("store", "mark"), "mark sources for a new download in full", None),
    "store/copy-out": (("store", "copy-out"), "a model folder the store keeps, into a new folder of this machine",
                       None),
    "agent/lock-image": (("agent", "lock-image"), "pin an image's base, archive date and packages in its "
                         "lock.json (needs Docker)", None),
    "agent/build-image": (("agent", "build-image"), "build an image from its lock.json on this machine (needs "
                          "Docker)", None),
    "train": (("train",), "train a model recipe on the dataset built from it", None),
    "eval": (("eval",), "run tests on a trained model, into its model card", None),
    "scaling": (("scaling",), "fit the scaling curve and predict a loss", None),
    "chat": (("chat",), "print a model folder's licenses and sources (--about); every other use of `wpt chat` is "
             "typed by a person", None),
    "manager": (("manager",), "download, build and train a model recipe in one command", None),
    "manager/stop": (("manager", "stop"), "stop the run a manager runs now: its checkpoint saved, its machines "
                     "given back", None),
    "doctor": (("doctor",), "check this machine: what is missing and how to fix it", None),
}
ABOUT = "--about"

JOB = {"type": "object", "required": ["id", "command", "started", "ended", "state", "exit_code"],
       "properties": {"id": STRING, "command": {"type": "string", "description": "the command line, as typed"},
                      "started": STRING, "ended": {"type": ["string", "null"]},
                      "state": {"enum": ["running", "succeeded", "failed"]},
                      "exit_code": {"type": ["integer", "null"], "description": "null while it runs; negative: "
                                    "ended by that signal"}}}
LINES = {"type": "object", "required": ["first", "next", "lines"],
         "properties": {"first": {"type": "integer", "description": "the number of the first line returned"},
                        "next": {"type": "integer", "description": "the number to ask for next"},
                        "lines": {"type": "array", "items": {
                            "type": "object", "required": ["stream", "text"],
                            "properties": {"stream": {"enum": ["stdout", "stderr"]}, "text": STRING}}}}}


def body(name, words, extra):
    """The JSON Schema of the body that starts a job: the arguments that follow the subcommand."""
    items = {"type": "string"}
    if name == "chat":  # no one types to a job: only the use that prints is allowed
        return {"type": "object", "required": ["args"], "additionalProperties": False, "properties": {
            "args": {"type": "array", "items": items, "contains": {"const": ABOUT},
                     "description": f"the arguments of `wpt chat`, which must hold {ABOUT}"}}}
    return {"type": "object", "additionalProperties": False, "properties": {
        "args": {"type": "array", "items": {**items, **(extra or {})}, "default": [],
                 "description": f"the arguments that follow `wpt {' '.join(words)}`, one string each "
                                f"(`wpt {' '.join(words)} --help`)"}}}


def routes(context, jobs: Jobs):
    @refusals
    def listed(values, body):
        return 200, [job.view() for job in jobs.all()]

    @refusals
    def one(values, body):
        return 200, jobs.get(values["id"]).view()

    @refusals
    def printed(values, body):
        if not values["first"].isdigit():
            return 404, {"problems": [f"no line {values['first']!r}: a number from 0"]}
        return 200, jobs.get(values["id"]).lines(int(values["first"]))

    @refusals
    def records(values, body):
        return 200, {"records": jobs.records(jobs.get(values["id"]), context.store)}

    def starting(words):
        def start(values, body):
            try:
                return 200, jobs.start(words, (body or {}).get("args", [])).view()
            except OSError as error:
                return 500, {"problems": [f"the job could not start: {error}"]}
        return start

    return [
        *[Route("POST", f"/api/jobs/{name}", starting(words), f"start `wpt {' '.join(words)}` as a job: {does}",
                body=body(name, words, extra), answer=JOB) for name, (words, does, extra) in JOBS.items()],
        Route("GET", "/api/jobs", listed, "every job this server started, oldest first (a stopped server forgets "
              "its jobs; their run records stay in the store)", answer={"type": "array", "items": JOB}),
        Route("GET", "/api/jobs/{id}", one, "one job: its command line, state and exit code", answer=JOB),
        Route("GET", "/api/jobs/{id}/lines/{first}", printed,
              "what the job printed to stdout and stderr from line number `first` on (the first printed is 0; ask "
              "again from `next` to follow it; a job keeps its last 20000 lines)", answer=LINES),
        Route("GET", "/api/jobs/{id}/records", records,
              "the run records in the store that name this job (start.json, ids.json, end.json of each; none when "
              "the subcommand writes no record)",
              answer={"type": "object", "required": ["records"], "properties": {"records": {
                  "type": "array", "items": {"type": "object", "required": ["folder", "start", "ids", "end"],
                                             "properties": {"folder": STRING, "start": {"type": "object"},
                                                            "ids": {"type": ["object", "null"]},
                                                            "end": {"type": ["object", "null"]}}}}}}),
    ]
