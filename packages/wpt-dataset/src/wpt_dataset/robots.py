"""A site's robots.txt can refuse a source; nothing overrides it.

Before any download from an http(s) address (main or mirror), the site's
robots.txt is read (RFC 9309, through protego). The source is refused when
the rules forbid the address to

- every crawler (the `*` group),
- WPT's own fetcher (`wpt-dataset`), or
- any crawler in TRAINING_CRAWLERS, those whose operator documents that
  they gather data to train AI models.

Crawlers that fetch a page for a person using an AI assistant (such as
Claude-User or ChatGPT-User) are not in the list: a site that refuses only
them has not refused training.

Only an answer saying the file does not exist (404 Not Found, 410 Gone)
means no rules: some sites publish no policy. Any other answer that is not
the file (another 4xx such as 401, 403 or 429, a 5xx, a timeout, an
unreachable site) makes that address unusable (RobotsUnreachable, an
OSError), since its policy cannot be known. A refusal or an unusable
address speaks only for that site: the source's next mirror is tried, with
its own robots.txt. Addresses that are not http(s) (local folders, ssh)
have no robots.txt.
"""

import urllib.error
import urllib.request
from urllib.parse import urlparse, urlunparse

from protego import Protego

from . import retry, sites
from .errors import CrawlForbidden

WPT_AGENT = "wpt-dataset"
MAX_BYTES = 500 * 1024  # the standard asks crawlers to read at least 500 KiB
NO_FILE = (404, 410)  # the only answers that mean "this site has no robots.txt"

# token -> where its operator states that it gathers AI training data. Only
# a first-party statement adds a crawler here; read 2026-10-14.
TRAINING_CRAWLERS: dict[str, str] = {
    "GPTBot": "https://developers.openai.com/api/docs/bots",  # "may be used in training our generative AI foundation models"
    "ClaudeBot": "https://support.claude.com/en/articles/8896518",  # "could potentially contribute to their training"
    "CCBot": "https://commoncrawl.org/about",  # "a primary training data source for many large language models"
    # "may be used for training future generations of Gemini models"
    "Google-Extended": "https://developers.google.com/search/docs/crawling-indexing/google-common-crawlers",
    "Applebot-Extended": "https://support.apple.com/en-us/119829",  # "used to train Apple's general purpose foundation models"
    # "crawls the web for use cases such as training foundation AI models"
    "meta-externalagent": "https://developers.facebook.com/docs/sharing/webmasters/web-crawlers/",
    "Amazonbot": "https://developer.amazon.com/amazonbot",  # "may be used to train Amazon AI models"
    "AI2Bot": "https://allenai.org/crawler",  # "This web content is used to train open language models"
    "MistralAI-Training": "https://docs.mistral.ai/robots",  # "datasets for training Mistral generative AI models"
}


class RobotsUnreachable(OSError):
    """The robots.txt could not be read: the address cannot be used."""


def check(address: str, kind: str, user_agent: str, cache: dict | None = None):
    """Raise CrawlForbidden when the robots.txt of address's site forbids
    fetching it. The robots.txt is asked at the host and port alone: a user
    name in the address is no part of the site's address (protego compares
    only path and query, so the checked addresses keep it). For git, the two
    paths a download asks for are checked too. cache: {robots.txt address:
    its rules}, filled as each site's file is read, for a program that
    fetches many addresses of one site (each read can count against a
    site's limit of requests); a file that could not be read is asked
    again."""
    parsed = urlparse(address)
    host = parsed.netloc.rpartition("@")[2]
    if parsed.scheme not in ("http", "https"):
        return
    urls = [address]
    if kind == "git":
        base = address.rstrip("/")
        urls += [f"{base}/info/refs?service=git-upload-pack", f"{base}/git-upload-pack"]
    robots_url = urlunparse((parsed.scheme, host, "/robots.txt", "", "", ""))
    rules = cache.get(robots_url) if cache is not None else None
    if rules is None:
        rules = _rules(robots_url, user_agent)
        if cache is not None:
            cache[robots_url] = rules
    for agent in ("*", WPT_AGENT, *TRAINING_CRAWLERS):
        for url in urls:
            if not rules.can_fetch(url, agent):
                who = "every crawler" if agent == "*" else agent
                raise CrawlForbidden(f"{address}: the site's robots.txt forbids {url} to {who}")


def _rules(robots_url: str, user_agent: str) -> Protego:
    request = urllib.request.Request(robots_url, headers={"User-Agent": user_agent})
    try:
        text = retry.call(lambda: _read(request), robots_url)
    except urllib.error.HTTPError as error:
        if error.code in NO_FILE:
            return Protego.parse("")  # no robots.txt: no rules
        raise RobotsUnreachable(f"{robots_url}: answered {error.code}, so the address cannot be used") from None
    except (urllib.error.URLError, OSError) as error:
        raise RobotsUnreachable(f"{robots_url}: unreachable ({error}), so the address cannot be used") from None
    return Protego.parse(text)


def _read(request) -> str:
    sites.paced(request.full_url)
    with urllib.request.urlopen(request, timeout=60) as response:  # follows redirects
        return response.read(MAX_BYTES).decode("utf-8", errors="replace")
