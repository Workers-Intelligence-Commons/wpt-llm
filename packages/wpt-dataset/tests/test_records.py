import gzip
import io
import json
import zipfile

import pytest

from wpt_dataset import markdown
from wpt_dataset.build import gather
from wpt_dataset.errors import InvalidFile, LicenseMapError
from wpt_dataset.manifest import FILE, Entry
from wpt_dataset.records import Skipped, joined_file, records
from wpt_dataset.sources import source_tree


def records_source(kit, repo, store, http_folder, licenses, keep=None):
    folder, base = http_folder
    lines = [{"id": "a", "text": "first words", "metadata": {"license": "CC-BY", "url": "https://example.org/a"}},
             {"id": "b", "text": "second words", "metadata": {"license": "Various open data", "url": "https://example.org/b"}},
             # some datasets store their metadata as a string holding JSON
             {"id": "c", "text": "third words", "metadata": json.dumps({"license": "CC0", "url": "https://example.org/c"})}]
    files = {"data/part-0.jsonl.gz": gzip.compress("".join(json.dumps(r) + "\n" for r in lines).encode()),
             "data/part-1.parquet": kit.parquet_bytes([
                 {"id": "d", "text": "fourth words", "metadata": {"license": "CC0", "url": "https://example.org/d"}},
                 {"id": "e", "text": "", "metadata": {"license": "CC0", "url": "https://example.org/e"}}])}
    (folder / "dataset.tar.gz").write_bytes(kit.tar_bytes(top="dataset", files=files, executable=set(), symlinks={}))
    record = {"format": "records", "paths": ["data/**"], "text": "text", "id": "id", "credit": ["metadata.url"]} | ({"keep": keep} if keep else {})
    return kit.add_source(repo, store, "example", "records", {
        "type": "http", "address": f"{base}/dataset.tar.gz", "extract": {"strip": 1}, "formats": [record], "licenses": licenses})


def test_only_records_whose_license_value_is_listed_become_documents_with_their_credit(kit, repo, store, http_folder):
    # each record of a large dataset carries its own license: WPT keeps a
    # record only when a license entry lists its value, and every model
    # trained on it can name each document's credit
    sid = records_source(kit, repo, store, http_folder, [
        kit.entry(["data/**"], license="CC-BY-4.0", place={"field": "metadata.license"}, expected=["CC-BY"]),
        kit.entry(["data/**"], license="CC0-1.0", place={"field": "metadata.license"}, expected=["CC0"])])

    documents, info, _ = gather(repo, store, log=lambda _: None)

    assert [d.id for d in documents] == [f"{sid}/data/part-0.jsonl.gz#a", f"{sid}/data/part-0.jsonl.gz#c",
                                         f"{sid}/data/part-1.parquet#d"]
    assert store.get_bytes(documents[0].sha256) == b"first words\n"
    credits = [json.loads(line) for line in store.get_bytes(info[sid]["credits"]).decode().splitlines()]
    assert credits == [
        {"document": f"{sid}/data/part-0.jsonl.gz#a", "license": "CC-BY-4.0", "credit": {"metadata.url": "https://example.org/a"}},
        {"document": f"{sid}/data/part-0.jsonl.gz#c", "license": "CC0-1.0", "credit": {"metadata.url": "https://example.org/c"}},
        {"document": f"{sid}/data/part-1.parquet#d", "license": "CC0-1.0", "credit": {"metadata.url": "https://example.org/d"}}]


def test_a_record_of_an_entry_that_does_not_train_is_left_out_alone_and_counted(kit, repo, store, http_folder):
    # an entry whose license is not admitted keeps its records out of
    # training, each alone: the other records of the same files still train,
    # and the dataset's record counts the values left out, per file
    under_review = {"decision": "under-review", "by": "Test Reviewer", "date": "2026-10-27"}
    sid = records_source(kit, repo, store, http_folder, [
        kit.entry(["data/**"], license="CC-BY-4.0", place={"field": "metadata.license"}, expected=["CC-BY"]),
        kit.entry(["data/**"], license="CC0-1.0", place={"field": "metadata.license"}, expected=["CC0"],
                  decision=under_review)])

    documents, info, _ = gather(repo, store, log=lambda _: None)

    assert [d.id for d in documents] == [f"{sid}/data/part-0.jsonl.gz#a"]
    check = info[sid]["check"]
    assert check["records"] == {"data/part-0.jsonl.gz": {"unlisted": {"Various open data": 1}, "untrained": {"CC0": 1}},
                                "data/part-1.parquet": {"untrained": {"CC0": 2}}}
    assert check["left_out"] == [{"why": "entry", "entry": 1, "license": "CC0-1.0", "files": 0,
                                  "detail": "the entry is under-review"}]


def test_keep_rules_leave_out_records_whose_field_holds_another_value(kit, repo, store, http_folder):
    # a dataset can label a whole file with one license although part of it
    # comes from a publisher that license does not cover; another field of
    # the record then names the part to keep
    sid = records_source(kit, repo, store, http_folder, [
        kit.entry(["data/**"], license="CC0-1.0", place={"field": "metadata.license"}, expected=["CC0"])],
        keep=[{"field": "metadata.url", "in": ["https://example.org/d", "https://example.org/e"]}])

    documents, _, _ = gather(repo, store, log=lambda _: None)

    assert [d.id for d in documents] == [f"{sid}/data/part-1.parquet#d"]


def test_the_entries_of_a_records_file_select_its_records_by_one_field(kit, repo, store, http_folder):
    # beside an entry selecting records by a field, an entry with no field place would let every other record
    # of the file in, whatever license the record itself carries
    sid = records_source(kit, repo, store, http_folder, [
        kit.entry(["data/**"], license="CC-BY-4.0", place={"field": "metadata.license"}, expected=["CC-BY"]),
        kit.entry(["data/**"], license="CC0-1.0")])

    with pytest.raises(LicenseMapError, match="every license entry covering it has the same field place"):
        source_tree(repo.source(sid), store)


def pages_source(kit, repo, store, http_folder, files, fields):
    """A source of JSON Lines files of web pages ({name: rows}), one page per record, all under CC0."""
    folder, base = http_folder
    (folder / "pages.tar.gz").write_bytes(kit.tar_bytes(top="pages", executable=set(), symlinks={}, files={
        name: "".join(json.dumps(row | {"license": [{"identifier": "CC0-1.0"}]}) + "\n" for row in rows).encode()
        for name, rows in files.items()}))
    return kit.add_source(repo, store, "example", "pages", {
        "type": "http", "address": f"{base}/pages.tar.gz", "extract": {"strip": 1},
        "licenses": [kit.entry(["*.ndjson"], license="CC0-1.0", place={"field": "license"},
                               expected=['[{"identifier": "CC0-1.0"}]'])],
        "formats": [{"format": "records", "paths": ["*.ndjson"], "text": "html", "as": "html", "id": "identifier"}
                    | fields]})


def test_a_record_holding_a_web_page_becomes_its_title_then_the_markdown_of_the_page(kit, repo, store,
                                                                                    http_folder):
    # a dataset of rendered web pages whose title is a field of its own:
    # the document starts with that title, then the page's Markdown, as an
    # html file of the page gives it, never its markup
    page = ('<!DOCTYPE html>\n<html><head><title>Ignored</title><style>p {}</style></head><body>'
            '<p>Some <i>words</i>.</p><figure><img src="m.png" alt="A map"/></figure>'
            '<script>alert(1)</script></body></html>')
    sid = pages_source(kit, repo, store, http_folder, {"pages_0.ndjson": [{"identifier": 7, "name": "Page one", "html": page}]},
                       {"title": "name"})

    documents, _, _ = gather(repo, store, log=lambda _: None)

    assert [d.id for d in documents] == [f"{sid}/pages_0.ndjson#7"]
    assert store.get_bytes(documents[0].sha256) == b"# Page one\n\nSome *words*.\n\n[image: A map]\n"


def test_of_several_revisions_of_one_record_only_the_newest_enters(kit, repo, store, http_folder):
    # a dataset may hold a document again each time it changed, in any
    # order and in any of its files: only its latest revision is training
    # text, and of two revisions with the same date, the later one
    sid = pages_source(kit, repo, store, http_folder, {
        "pages_0.ndjson": [{"identifier": 7, "date_modified": "2025-03-02T10:00:00Z", "html": "<p>7 newer</p>"},
                           {"identifier": 8, "date_modified": "2025-01-01T00:00:00Z", "html": "<p>8 only</p>"},
                           {"identifier": 7, "date_modified": "2025-03-01T10:00:00Z", "html": "<p>7 older</p>"},
                           {"identifier": 9, "date_modified": "2025-01-01T00:00:00Z", "html": "<p>9 older</p>"},
                           {"identifier": 10, "date_modified": "2025-04-01T00:00:00Z", "html": "<p>10 first</p>"},
                           {"identifier": 10, "date_modified": "2025-04-01T00:00:00Z", "html": "<p>10 second</p>"}],
        "pages_1.ndjson": [{"identifier": 7, "date_modified": "2025-02-01T00:00:00Z", "html": "<p>7 oldest</p>"},
                           {"identifier": 9, "date_modified": "2025-02-01T00:00:00Z", "html": "<p>9 newer</p>"}]},
        {"newest": "date_modified"})

    documents, _, _ = gather(repo, store, log=lambda _: None)

    assert {d.id: store.get_bytes(d.sha256) for d in documents} == {
        f"{sid}/pages_0.ndjson#7": b"7 newer\n", f"{sid}/pages_0.ndjson#8": b"8 only\n",
        f"{sid}/pages_0.ndjson#10": b"10 second\n", f"{sid}/pages_1.ndjson#9": b"9 newer\n"}


def scan(date, articles):
    """A newspaper page scan: one JSON document whose "full articles"
    holds its articles."""
    return json.dumps({"lccn": {"title": "The town paper"}, "edition": {"date": date}, "page_number": 1,
                       "bboxes": [], "full articles": [
                           {"id": f"{number}_{date}_p1.json", "full_article_id": str(number), "headline": headline,
                            "byline": "", "article": article}
                           for number, (headline, article) in enumerate(articles, start=1)]}, indent=4).encode()


def test_records_listed_in_one_json_document_take_the_license_of_their_file(kit, repo, store, http_folder):
    # a dataset that states one license for all its records, each page scan
    # one JSON document listing its articles
    folder, base = http_folder
    (folder / "paper_1770.tar.gz").write_bytes(kit.tar_bytes(top="paper_1770", executable=set(), symlinks={}, files={
        "1770-08-23_p1.json": scan("1770-08-23", [("BOSTON", "first words"), ("", "")]),
        "1770-08-30_p1.json": scan("1770-08-30", [("LONDON", "second words")])}))
    sid = kit.add_source(repo, store, "example", "scans", {
        "type": "http", "address": f"{base}/paper_1770.tar.gz", "licenses": [kit.entry(["paper_1770/**"], license="LicenseRef-public-domain", **kit.ITSELF, text="Published in the United States in 1770.")],
        "formats": [{"format": "records", "paths": ["paper_*/*.json"], "records": "full articles", "text": "article",
                     "id": "id", "credit": ["headline"]}]})

    documents, info, _ = gather(repo, store, log=lambda _: None)

    first, second = f"{sid}/paper_1770/1770-08-23_p1.json#1_1770-08-23_p1.json", \
        f"{sid}/paper_1770/1770-08-30_p1.json#1_1770-08-30_p1.json"
    assert [d.id for d in documents] == [first, second]
    assert store.get_bytes(documents[1].sha256) == b"second words\n"
    credits = [json.loads(line) for line in store.get_bytes(info[sid]["credits"]).decode().splitlines()]
    assert credits == [{"document": first, "license": "LicenseRef-public-domain", "credit": {"headline": "BOSTON"}},
                       {"document": second, "license": "LicenseRef-public-domain", "credit": {"headline": "LONDON"}}]


NAMESPACES = {"rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#", "edm": "http://www.europeana.eu/schemas/edm/",
             "dc": "http://purl.org/dc/elements/1.1/", "dcterms": "http://purl.org/dc/terms/"}
PUBLIC_DOMAIN_MARK = "http://creativecommons.org/publicdomain/mark/1.0/"
ISSUE_FIELDS = {
    "format": "records", "paths": ["fulltext/**"], "namespaces": NAMESPACES, "text": "//edm:FullTextResource/rdf:value",
    "credit": ["$metadata//edm:ProvidedCHO/dc:title", "$metadata//edm:dataProvider"],
    "keep": [{"field": "number(substring($metadata//edm:ProvidedCHO/dcterms:issued, 1, 4)) <= 1955", "in": ["true"]}],
    "join": {"name": "metadata", "paths": ["metadata/**"], "key": "BibliographicResource_([0-9]+)\\.xml$",
             "joined_key": "BibliographicResource%2F([0-9]+)\\.edm\\.xml$"}}


def fulltext(number, text):
    """An issue's full text as a library publishes it: entities its own
    DTD declares, used in attributes."""
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE RDF [\n<!ENTITY text '
            f'"http://example.org/fulltext/9200396/BibliographicResource_{number}/0" >\n]>\n'
            f'<rdf:RDF xmlns:rdf="{NAMESPACES["rdf"]}" xmlns:edm="{NAMESPACES["edm"]}" xmlns:dc="{NAMESPACES["dc"]}">'
            f'<edm:FullTextResource rdf:about="&text;"><dc:language>fr</dc:language>'
            f'<rdf:value>{text}</rdf:value></edm:FullTextResource></rdf:RDF>').encode()


def metadata(number, issued, rights):
    about = f"http://data.theeuropeanlibrary.org/BibliographicResource/{number}"
    return (f'<?xml version="1.0" encoding="UTF-8"?><rdf:RDF xmlns:rdf="{NAMESPACES["rdf"]}" '
            f'xmlns:edm="{NAMESPACES["edm"]}" xmlns:dc="{NAMESPACES["dc"]}" xmlns:dcterms="{NAMESPACES["dcterms"]}" '
            f'xmlns:ore="http://www.openarchives.org/ore/terms/"><edm:ProvidedCHO rdf:about="{about}">'
            f'<dcterms:issued>{issued}</dcterms:issued><dc:title>Journal - {issued}</dc:title></edm:ProvidedCHO>'
            f'<ore:Aggregation rdf:about="{about}#aggregation"><edm:dataProvider>A national library'
            f'</edm:dataProvider><edm:rights rdf:resource="{rights}"/></ore:Aggregation></rdf:RDF>').encode()


def zip_bytes(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path, content in files.items():
            info = zipfile.ZipInfo(path)
            info.create_system = 0  # made on MS-DOS, with no unix mode, as some publishers' are
            archive.writestr(info, content)
    return buffer.getvalue()


def test_an_issue_takes_its_text_from_one_file_and_its_license_and_credit_from_the_joined_one(kit, repo, store,
                                                                                               http_folder):
    # newspaper issues published as full texts in one archive and their
    # metadata (rights, date, title, library) in another; an issue enters
    # only when its own metadata says public domain and the `keep` rule on
    # its date holds
    folder, base = http_folder
    (folder / "fulltext.zip").write_bytes(zip_bytes({
        f"9200396/BibliographicResource_{number}.xml": fulltext(number, f"JOURNAL\nissue {number}")
        for number in (1, 2, 3, 4)}))
    metadata_file = "http%3A%2F%2Fdata.theeuropeanlibrary.org%2FBibliographicResource%2F{}.edm.xml".format
    (folder / "metadata.zip").write_bytes(zip_bytes({
        metadata_file(1): metadata(1, "1788-02-01", PUBLIC_DOMAIN_MARK),
        metadata_file(2): metadata(2, "1960-05-01", PUBLIC_DOMAIN_MARK),  # too recent
        metadata_file(4): metadata(4, "1900-01-01", "http://rightsstatements.org/vocab/InC/1.0/")}))  # 3 has none
    sid = kit.add_source(repo, store, "example", "library", {
        "type": "http", "address": f"{base}/fulltext.zip", "addresses": [{"folder": "fulltext", "address": f"{base}/fulltext.zip"},
                      {"folder": "metadata", "address": f"{base}/metadata.zip"}],
        "licenses": [kit.entry(["fulltext/**", "metadata/**"], license="LicenseRef-public-domain",
                               place={"field": "$metadata//edm:rights/@rdf:resource"}, expected=[PUBLIC_DOMAIN_MARK],
                               text="The publication's own public domain mark.")],
        "formats": [ISSUE_FIELDS]})

    documents, info, _ = gather(repo, store, log=lambda _: None)

    issue = f"{sid}/fulltext/9200396/BibliographicResource_1.xml#1"
    assert [d.id for d in documents] == [issue]
    assert store.get_bytes(documents[0].sha256) == markdown.text("JOURNAL\nissue 1")
    credits = [json.loads(line) for line in store.get_bytes(info[sid]["credits"]).decode().splitlines()]
    assert credits == [{"document": issue, "license": "LicenseRef-public-domain",
                        "credit": {"$metadata//edm:ProvidedCHO/dc:title": "Journal - 1788-02-01",
                                   "$metadata//edm:dataProvider": "A national library"}}]


def test_a_fulltext_file_finds_the_metadata_file_that_joins_it():
    tree = {"fulltext/9200396/BibliographicResource_7.xml": Entry(FILE, "a" * 64),
            "metadata/BibliographicResource%2F7.edm.xml": Entry(FILE, "b" * 64),
            "metadata/BibliographicResource%2F8.edm.xml": Entry(FILE, "c" * 64)}

    assert joined_file(ISSUE_FIELDS, "fulltext/9200396/BibliographicResource_7.xml", tree, {}) == tree["metadata/BibliographicResource%2F7.edm.xml"]
    assert joined_file(ISSUE_FIELDS, "fulltext/9200396/BibliographicResource_9.xml", tree, {}) is None


def test_an_xml_record_never_reads_an_external_entity(tmp_path):
    # the records come from upstream: an entity naming a file or an address
    # outside the file must not bring that content in
    secret = tmp_path / "secret.txt"
    secret.write_text("SECRET", encoding="utf-8", newline="\n")
    issue = tmp_path / "issue.xml"
    issue.write_bytes(fulltext(1, "words").replace(
        b"<!ENTITY text", f'<!ENTITY secret SYSTEM "{secret.as_uri()}">\n<!ENTITY text'.encode())
        .replace(b"<rdf:value>words", b"<rdf:value>&secret;"))
    entry = {key: value for key, value in ISSUE_FIELDS.items() if key not in ("keep", "join")}

    with pytest.raises(InvalidFile) as refused:
        list(records(issue, "issue.xml", entry, None, {None: "LicenseRef-public-domain"}, Skipped({})))
    assert "SECRET" not in str(refused.value)
