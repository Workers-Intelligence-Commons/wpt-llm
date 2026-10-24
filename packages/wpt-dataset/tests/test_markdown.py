import gzip
import json

import pytest

from wpt_dataset import conversations
from wpt_dataset.errors import FormatMapError, InvalidFile, NotText
from wpt_dataset.cli import main
from wpt_dataset.edit import unpack, write_patch
from wpt_dataset.encoding import decode
from wpt_dataset.markdown import documents
from wpt_dataset.markdown_check import problems
from wpt_dataset.sources import source_tree

HEAD = ('<?xml version="1.0" encoding="utf-8"?>\n<html xmlns="http://www.w3.org/1999/xhtml" '
        'xmlns:epub="http://www.idpf.org/2007/ops"><head><title>Page</title></head>')
PAGE = HEAD + """<body><section><h2>The Prison</h2>
<p>Socrates is <i>calm</i> and *says* 1. things.<a href="notes.xhtml#note-1" epub:type="noteref">1</a></p>
<table><tbody><tr><td epub:type="z3998:persona">Crito</td>
<td>Why so early?<a href="notes.xhtml#note-9" epub:type="noteref">9</a></td></tr></tbody></table>
<p>A line<br/>and another</p>
<figure><img src="../images/map.png" alt="Athens,  the  prison"/> and
<svg xmlns="http://www.w3.org/2000/svg"><title>Seal</title></svg> <img src="../images/rule.png" alt=""/><figcaption>The city</figcaption></figure></section></body></html>"""
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
    return {c.path: c.markdown for c in documents(source, source_tree(source, store)[0], store, repo)}


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
        "[image: Athens, the prison] and [image: Seal] [image]\n\n"  # where each image was, with its alt text
        "The city\n\n"
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
    assert [(c.path, c.markdown) for c in documents(source, source_tree(source, store)[0], store, repo)] == [
        ("run.py", b"```python path=\"run.py\"\nprint(1)\n```\n")]


def test_a_file_without_format_or_with_broken_markup_stops_the_build(kit, repo, store, tmp_path):
    with pytest.raises(FormatMapError, match="extra.dat matches no format entry"):
        markdown_of(kit, repo, store, tmp_path, {"plain.txt": b"words\n", "extra.dat": b"x\n"})
    with pytest.raises(NotText, match="text/one.xhtml is not well-formed XHTML"):
        markdown_of(kit, repo, store, tmp_path / "second", {"text/one.xhtml": PAGE.replace("</p>", "", 1).encode()})


GUTENBERG = """<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"
    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta http-equiv="Content-Type" content="text/html;charset=iso-8859-1" />
<title>Les Misérables</title>
<style type="text/css">p { text-indent: 2% }</style>
</head>
<body>
<p class="center"><i>par<br>
Victor Hugo<br></i></p>
<h2>LIVRE PREMIER</h2>
<h3>Un juste</h3>
<p><span class="pagenum"><a id="Page_5">{5}</a></span>En 1815, M. Myriel était évêque de
<i>Digne</i>.<a name="FNanchor_1" id="FNanchor_1"></a><a href="#Footnote_1" class="fnanchor">[1]</a>
<p>C'était un vieillard d'environ soixante-quinze ans.<a name="FNanchor_2" id="FNanchor_2"></a><a
href="#Footnote_2" class="fnanchor">[2]</a>
<div class="figcenter"><img src="images/myriel.jpg" alt="L'évêque
  Myriel"><p class="caption">À sa table.</p></div>
<p><img src="images/deco.png">Fin du chapitre.<a name="FNanchor_3" id="FNanchor_3"></a><a href="#Footnote_3"
class="fnanchor">[1]</a>
<div class="footnote"><p><a name="Footnote_1" id="Footnote_1"></a><a href="#FNanchor_1"><span
class="label">[1]</span></a> Digne est en Provence.</div>
<div class="footnote"><p><a name="Footnote_2" id="Footnote_2"></a><a href="#FNanchor_2"><span
class="label">[2]</span></a> Il était né en 1740.
<p><a name="Footnote_3" id="Footnote_3"></a><a href="#FNanchor_3"><span
class="label">[1]</span></a> Fin.</div>
<p class="footnote">[4] Une note que rien n'appelle.</p>
</body>
</html>
"""


def test_a_latin_1_gutenberg_page_becomes_utf_8_markdown_and_takes_a_removal_patch(kit, repo, store, tmp_path,
                                                                                  http_folder):
    # Project Gutenberg's pages are HTML, not XML, and older ones are Latin-1 as their
    # <meta> states: the page must read as written, its images leave markers, its
    # footnotes become Markdown notes, its page numbers stay, and a removal patch
    # made from the unpacked page must apply to its own bytes
    folder, base = http_folder
    (folder / "17489-h.htm").write_bytes(GUTENBERG.encode("latin-1"))
    sid = kit.add_source(repo, store, "gutenberg", "17489", {
        "type": "http-file", "address": f"{base}/17489-h.htm", "revision": "2026-10-23",
        "formats": [{"format": "html", "paths": ["**"]}]})

    def markdown():
        source = repo.source(sid)
        return documents(source, source_tree(source, store)[0], store, repo)

    [page] = markdown()
    assert page.markdown.decode("utf-8") == (
        "*par\\\nVictor Hugo*\n\n"  # the line break ending the italics stays outside its marks
        "## LIVRE PREMIER\n\n"
        "### Un juste\n\n"
        "{5}En 1815, M. Myriel était évêque de *Digne*.[^1]\n\n"
        "C'était un vieillard d'environ soixante-quinze ans.[^2]\n\n"
        "[image: L'évêque Myriel]\n\n"
        "À sa table.\n\n"
        "[image]Fin du chapitre.[^1-2]\n\n"  # a label met again gets a number
        "\\[4\\] Une note que rien n'appelle.\n\n"  # no link refers to it: ordinary text, as printed
        "[^1]: Digne est en Provence.\n\n"
        "[^2]: Il était né en 1740.\n\n"  # two notes in one footnote element stay two notes
        "[^1-2]: Fin.\n")
    assert problems(page.markdown.decode("utf-8")) == []

    edit = tmp_path / "edit"
    unpack(repo, repo.source(sid), store, edit)
    kept = [line for line in (edit / "17489-h.htm").read_bytes().split(b"\n") if b"vieillard" not in line]
    (edit / "17489-h.htm").write_bytes(b"\n".join(kept))
    write_patch(repo, repo.source(sid), store, edit, "remove-a-line")
    assert b"-<p>C'\xe9tait un vieillard" in repo.source(sid).patches[0].data  # the page's own Latin-1 bytes
    [page] = markdown()
    assert "vieillard" not in page.markdown.decode("utf-8") and "évêque de *Digne*" in page.markdown.decode("utf-8")


def test_an_html_page_with_tags_xml_refuses_and_odd_list_starts_still_converts(kit, repo, store, tmp_path):
    # the html format never refuses a page: a tag name with a colon is one more
    # container, and a list's start is read as HTML reads it, 1 when it holds no number
    page = (b'<html><body><p>Script<mw:editTemplateData page="x"></mw:editTemplateData> forms.</p>'
            b'<ol start=""><li>One</li><li>Two</li></ol><ol start=" 7th"><li>Seven</li></ol></body></html>')
    found = markdown_of(kit, repo, store, tmp_path, {"page.html": page}, [{"format": "html", "paths": ["**"]}])
    assert found["page.html"].decode() == "Script forms.\n\n1. One\n2. Two\n\n7. Seven\n"


@pytest.mark.parametrize("data, entry, read", [
    (b"\xff\xfe" + "été".encode("utf-16-le"), {"format": "markdown"}, "\ufeffété"),
    (b"Title: X\r\nCharacter set encoding: ISO-Latin-1\r\n\r\n"
     b"*** START OF THE PROJECT GUTENBERG EBOOK X ***\r\n\xe9t\xe9\r\n", {"format": "text"}, "été"),
    (b"Character set encoding: ASCII\n*** START OF THIS PROJECT GUTENBERG EBOOK X ***\n\xe9t\xe9\n", {"format": "text"},
     "is not text in 'ASCII', the encoding it states (byte 78)"),
    (b'<?xml version="1.0" encoding="ISO-8859-1"?>\n<p>\xe9t\xe9</p>', {"format": "xhtml"}, "été"),
    (b'<?xml version="1.0" encoding="iso-8859-1"?>\n<html><p>\x93\xe9t\xe9\x94', {"format": "html"},
     "“été”"),  # no <meta>: the declaration's label is read as the prescan reads one
    (b'<!-- <meta charset="koi8-r"> --><meta charset="iso-8859-1"><p>\x93\xe9t\xe9\x94', {"format": "html"},
     "“été”"),  # the HTML standard reads iso-8859-1 as windows-1252
    (b'<meta charset="x-unknown"><p>\xe9t\xe9', {"format": "html"},
     "states the encoding 'x-unknown', which Python does not know"),
    (b"#!/usr/bin/env python\n# -*- coding: latin-1 -*-\ns = '\xe9t\xe9'\n", {"format": "code", "language": "python"},
     "s = 'été'"),
    (b"caf\xe9\nCharacter set encoding: ISO-8859-1\n", {"format": "text"}, "is not UTF-8 text"),  # no Gutenberg header
])
def test_a_file_is_text_in_the_encoding_it_states_and_nothing_is_guessed(data, entry, read):
    # a stated encoding is the file's own word on its text; a file that states none
    # stays UTF-8 or no text, and a statement Python cannot follow is named
    try:
        assert read in decode(data, entry, "f")
    except NotText as error:
        assert str(error) == f"f {read}"


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


def test_a_chat_file_is_checked_and_never_becomes_a_pretraining_document(kit, repo, store, tmp_path):
    # conversations are for the post-training step named by their tags: they
    # must reach no pre-training mixture, and a malformed one must stop the check
    good = (b'{"schema": 1, "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}, '
            b'{"role": "assistant", "content": [{"type": "text", "text": "Hello"}]}]}\n')
    formats = [{"format": "chat", "paths": ["*.jsonl"]}, {"format": "text", "paths": ["plain.txt"]}]
    found = markdown_of(kit, repo, store, tmp_path / "good", {"talks.jsonl": good, "plain.txt": b"words\n"}, formats)
    assert list(found) == ["plain.txt"]

    thinking_user = good.replace(b'"assistant"', b'"user"').replace(b'"text", "text": "Hello"', b'"thinking", "text": "x"')
    with pytest.raises(InvalidFile, match="talks.jsonl line 2: at messages/1/content/0"):  # only the assistant thinks
        markdown_of(kit, repo, store, tmp_path / "bad", {"talks.jsonl": good + thinking_user}, formats)
    lost = good.replace(b'"role": "assistant", "content": [{"type": "text", "text": "Hello"}]',
                        b'"role": "tool", "content": [{"type": "tool_result", "call": "c9", "content": []}]')
    with pytest.raises(InvalidFile, match="tool_result answers 'c9', which no earlier tool_call made"):
        markdown_of(kit, repo, store, tmp_path / "lost", {"talks.jsonl": lost}, formats)


def test_a_message_tree_gives_every_human_path_as_a_conversation(kit, repo, store, http_folder):
    # datasets such as OpenAssistant store branching conversations as messages
    # naming their parent; post-training needs each path from the first message
    # to a last one, and nothing written by a model or flagged by its reviewers
    rows = [
        {"id": "q", "parent": None, "role": "prompter", "text": "Why is the sky blue?"},
        {"id": "a1", "parent": "q", "role": "assistant", "text": "Air scatters blue light most."},
        {"id": "a2", "parent": "q", "role": "assistant", "text": "Rayleigh scattering."},
        {"id": "f1", "parent": "a1", "role": "prompter", "text": "And at sunset?"},
        {"id": "f1a", "parent": "f1", "role": "assistant", "text": "The light crosses more air, so red is left."},
        {"id": "f2", "parent": "a1", "role": "prompter", "text": "Thanks!"},  # unanswered: left off
        {"id": "m", "parent": "a2", "role": "prompter", "text": "Say more."},
        {"id": "ma", "parent": "m", "role": "assistant", "text": "written by a model", "synthetic": True},
        {"id": "d", "parent": "q", "role": "assistant", "text": "removed", "deleted": True},
        {"id": "dd", "parent": "d", "role": "prompter", "text": "after a removed one"},
        {"id": "p", "parent": None, "role": "prompter", "text": "My phone number is ...", "labels": {"pii": 0.7}},
        {"id": "pa", "parent": "p", "role": "assistant", "text": "Noted."},
    ]
    data = gzip.compress("".join(json.dumps(row) + "\n" for row in rows).encode())
    tree = {"id": "id", "parent": "parent", "role": "role", "text": "text",
            "roles": {"prompter": "user", "assistant": "assistant"},
            "drop": [{"field": "synthetic", "is": True}, {"field": "deleted", "is": True},
                     {"field": "labels.pii", "above": 0}]}
    formats = [{"format": "chat", "paths": ["*.jsonl.gz"], "tree": tree}]
    folder, base = http_folder

    def tree_source(item, data):  # compressed, as OpenAssistant publishes it
        (folder / f"{item}.tar.gz").write_bytes(kit.tar_bytes(top=item, files={"messages.jsonl.gz": data},
                                                              executable=set(), symlinks={}))
        return kit.add_source(repo, store, "example", item, {
            "type": "http-archive", "address": f"{base}/{item}.tar.gz", "revision": "2026-10-16",
            "extract": {"strip": 1}, "formats": formats, "tags": [{"tags": ["sft"], "paths": ["**"]}]})

    sid = tree_source("talks", data)

    found, _ = conversations.read(repo, store, "sft")
    said = {cid: [m["content"][0]["text"] for m in c["messages"]] for cid, c in found}
    assert said == {
        f"{sid}/messages.jsonl.gz:f1a": ["Why is the sky blue?", "Air scatters blue light most.", "And at sunset?",
                                      "The light crosses more air, so red is left."],
        f"{sid}/messages.jsonl.gz:a1": ["Why is the sky blue?", "Air scatters blue light most."],
        f"{sid}/messages.jsonl.gz:a2": ["Why is the sky blue?", "Rayleigh scattering."],
    }
    assert [m["role"] for m in found[0][1]["messages"]] == ["user", "assistant", "user", "assistant"]

    rows.append({"id": "x", "parent": "gone", "role": "prompter", "text": "orphan"})
    data = gzip.compress("".join(json.dumps(row) + "\n" for row in rows).encode())
    source = repo.source(tree_source("orphan", data))
    with pytest.raises(InvalidFile, match="1 messages name a parent the file does not hold, such as gone"):
        list(documents(source, source_tree(source, store)[0], store, repo))
