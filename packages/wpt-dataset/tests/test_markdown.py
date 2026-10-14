import pytest

from wpt_dataset.errors import FormatMapError, NotText
from wpt_dataset.cli import main
from wpt_dataset.markdown import documents
from wpt_dataset.markdown_check import problems
from wpt_dataset.sources import source_tree

HEAD = ('<?xml version="1.0" encoding="utf-8"?>\n<html xmlns="http://www.w3.org/1999/xhtml" '
        'xmlns:epub="http://www.idpf.org/2007/ops"><head><title>Page</title></head>')
PAGE = HEAD + """<body><section><h2>The Prison</h2>
<p>Socrates is <i>calm</i> and *says* 1. things.<a href="notes.xhtml#note-1" epub:type="noteref">1</a></p>
<table><tbody><tr><td epub:type="z3998:persona">Crito</td>
<td>Why so early?<a href="notes.xhtml#note-9" epub:type="noteref">9</a></td></tr></tbody></table>
<p>A line<br/>and another</p></section></body></html>"""
NOTES = HEAD + """<body><section epub:type="endnotes"><h2>Endnotes</h2><ol><li id="note-1">
<p>See <i>Phaedo</i>. <a href="one.xhtml#noteref-1" epub:type="backlink">↩</a></p></li></ol></section></body></html>"""
FORMATS = [{"format": "xhtml", "paths": ["text/*.xhtml"]}, {"format": "text", "paths": ["plain.txt"]},
           {"format": "code", "language": "python", "paths": ["code/**"]}]


def markdown_of(kit, repo, store, tmp_path, files, formats=FORMATS):
    tmp_path.mkdir(exist_ok=True)
    address, commit = kit.git_project(tmp_path / "book", files=files, executable=set(), symlinks={})
    sid = kit.add_source(repo, store, "example", "book", {"type": "git", "address": address, "revision": commit,
                                                           "formats": formats})
    source = repo.source(sid)
    return dict(documents(source, source_tree(source, store)[0], store))


def test_every_format_becomes_markdown_that_reads_as_written(kit, repo, store, tmp_path):
    # the model trains on Markdown only, so each format must keep its words,
    # its structure and its notes, and show Markdown's own characters literally
    files = {"text/one.xhtml": PAGE.encode(), "text/notes.xhtml": NOTES.encode(),
             "plain.txt": b"# not a heading\nsecond *line* \n\nnew paragraph\n",
             "code/run.py": b"print('```')\n"}
    found = markdown_of(kit, repo, store, tmp_path, files)

    assert found["text/one.xhtml"].decode() == (
        "## The Prison\n\n"
        "Socrates is *calm* and \\*says\\* 1. things.[^1]\n\n"
        "**Crito:** Why so early?\n\n"  # note 9 is not in the source: its reference is dropped
        "A line\\\nand another\n\n"
        "[^1]: See *Phaedo*.\n")
    assert "text/notes.xhtml" not in found  # its only note now lives in the page that refers to it
    assert found["plain.txt"] == b"\\# not a heading\\\nsecond \\*line\\*\n\nnew paragraph\n"
    assert found["code/run.py"] == b"````python path=\"./code/run.py\"\nprint('```')\n````\n"  # from the repository's root


def test_code_outside_a_repository_names_its_path_without_a_root(kit, repo, store, http_folder):
    # "./" stands for a repository's root; a downloaded file or archive has none
    folder, base = http_folder
    (folder / "run.py").write_bytes(b"print(1)\n")
    sid = kit.add_source(repo, store, "example", "script", {
        "type": "http-file", "address": f"{base}/run.py", "revision": "2026-10-14",
        "formats": [{"format": "code", "language": "python", "paths": ["**"]}]})
    source = repo.source(sid)
    assert documents(source, source_tree(source, store)[0], store) == [
        ("run.py", b"```python path=\"run.py\"\nprint(1)\n```\n")]


def test_a_file_without_format_or_with_broken_markup_stops_the_build(kit, repo, store, tmp_path):
    with pytest.raises(FormatMapError, match="extra.dat matches no format entry"):
        markdown_of(kit, repo, store, tmp_path, {"plain.txt": b"words\n", "extra.dat": b"x\n"})
    with pytest.raises(NotText, match="text/one.xhtml is not well-formed XHTML"):
        markdown_of(kit, repo, store, tmp_path / "second", {"text/one.xhtml": PAGE.replace("</p>", "", 1).encode()})


def test_markdown_check_finds_broken_notes_and_leftover_tags():
    # a converter mistake on a new kind of input must show, not slip into training
    assert problems("Text.[^1] An escaped \\<i> and `<b>` in code.\n\n[^1]: The note.\n") == []
    assert problems("Text.[^2] and <span class=\"x\">more</span>\n\n[^3]: Lost.\n") == [
        "note [^2] has no definition", "note [^3] is defined but never referred to",
        "unescaped HTML tag '<span'", "unescaped HTML tag '</span>'"]


def test_markdown_check_command_passes_on_converted_pages(kit, repo, store, tmp_path, monkeypatch, capsys):
    files = {"text/one.xhtml": PAGE.encode(), "text/notes.xhtml": NOTES.encode(), "plain.txt": b"<b> stays text\n",
             "code/run.py": b"print('<b>')\n"}
    markdown_of(kit, repo, store, tmp_path, files)
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())
    assert main(["--root", str(repo.root), "markdown-check"]) == 0
    assert capsys.readouterr().out.endswith(": 2 documents checked, 1 notes, OK\n")  # the page and plain.txt
