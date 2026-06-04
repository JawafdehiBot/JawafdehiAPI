"""Service for searching and enriching CIAA cases with news articles."""

import concurrent.futures
import json
import logging
import os
import re
import tempfile
import time
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus, urlparse

import requests
from django.db import close_old_connections, transaction

from cases.models import (
    Case,
    CaseEntityRelationship,
    DocumentSource,
    RelationshipType,
    SourceType,
)

logger = logging.getLogger(__name__)

_MAX_HTML_REGEX_LENGTH = 500_000

_DEFAULT_LLM_MODEL = "gpt-4.5"
_DEFAULT_LLM_BASE_URL = "https://llm-proxy.jawafdehi.org/v1"

_HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; JawafdehiAPI/1.0)",
}
_FETCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; JawafdehiAPI/1.0)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en,ne;q=0.9",
}


def _truncate_for_regex(html: str) -> str:
    """Truncate HTML to a safe length for regex operations."""
    if len(html) > _MAX_HTML_REGEX_LENGTH:
        return html[:_MAX_HTML_REGEX_LENGTH]
    return html


_ALLOWED_HOSTS = frozenset({"ciaa.gov.np", "ngm-store.jawafdehi.org"})

_OFFICIAL_PRESS_RELEASE_PATTERNS = (
    re.compile(
        r"^https?://(?:www\.)?ciaa\.gov\.np/(?:index\.php/)?pressrelease/",
        re.IGNORECASE,
    ),
)

_URL_BLOCKLIST_PATTERNS = (
    re.compile(r"/tag[/?]|/category[/?]|/author[/?]|/page/\d+", re.IGNORECASE),
)

_NON_NEWS_DOMAIN_PATTERNS = (
    re.compile(r"^https?://(?:[a-z-]+\.)?wikipedia\.org/", re.IGNORECASE),
    re.compile(r"^https?://(?:[a-z-]+\.)?facebook\.com/", re.IGNORECASE),
)


def _is_official_press_release(url: str) -> bool:
    """Check if a URL is an official CIAA press release page (not third-party news)."""
    for pattern in _OFFICIAL_PRESS_RELEASE_PATTERNS:
        if pattern.search(url):
            return True
    return False


def _is_url_blocklisted(url: str) -> Optional[str]:
    """Check if a URL matches a blocklist pattern. Returns reason string or None."""
    for pattern in _URL_BLOCKLIST_PATTERNS:
        if pattern.search(url):
            return "tag/category/author page"
    for pattern in _NON_NEWS_DOMAIN_PATTERNS:
        if pattern.search(url):
            return "non-news domain (wikipedia/facebook)"
    return None


# Event types in a CIAA corruption case lifecycle
_EVENT_INVESTIGATION = "investigation"
_EVENT_FILING = "filing"
_EVENT_HEARING = "hearing"
_EVENT_VERDICT = "verdict"
_EVENT_APPEAL = "appeal"

_ALL_EVENT_TYPES = (
    _EVENT_INVESTIGATION,
    _EVENT_FILING,
    _EVENT_HEARING,
    _EVENT_VERDICT,
    _EVENT_APPEAL,
)

_EVENT_OTHER = "other"
_LIFECYCLE_EVENT_TYPES = frozenset(
    {
        _EVENT_INVESTIGATION,
        _EVENT_FILING,
        _EVENT_HEARING,
        _EVENT_VERDICT,
        _EVENT_APPEAL,
    }
)
_EVENT_LIFECYCLE_ORDER = {
    _EVENT_INVESTIGATION: 1,
    _EVENT_FILING: 2,
    _EVENT_HEARING: 3,
    _EVENT_VERDICT: 4,
    _EVENT_APPEAL: 5,
    _EVENT_OTHER: 6,
}

_MAX_ARTICLES_PER_EVENT_TYPE = 1
_QUERY_LIMIT = 15
_QUERY_RESERVED_ENGLISH_SLOTS = 4
_QUERY_RESERVED_EVENT_SLOTS = 4

_EVENT_QUERY_TEMPLATES: dict[str, list[str]] = {
    _EVENT_INVESTIGATION: [
        "{name} अख्तियार अनुसन्धान",
        "{name} CIAA investigation",
    ],
    _EVENT_FILING: [
        "{name} अख्तियार मुद्दा दायर",
        "{name} CIAA charge sheet special court",
    ],
    _EVENT_HEARING: [
        "{name} सुनुवाइ विशेष अदालत",
        "{name} hearing special court corruption",
    ],
    _EVENT_VERDICT: [
        "{name} फैसला विशेष अदालत",
        "{name} verdict special court corruption",
        "{name} ठहर भ्रष्टाचार",
    ],
    _EVENT_APPEAL: [
        "{name} पुनरावेदन सर्वोच्च",
        "{name} supreme court appeal corruption",
        "{name} अख्तियार पुनरावेदन",
        "{name} सर्वोच्च अदालत फैसला",
    ],
}

_VERIFY_SYSTEM_PROMPT = """\
You are a fact-checking assistant for a Nepal corruption accountability platform.
Your job is to determine whether a given news article is genuinely about the same
CIAA Special Court corruption case as the case described below.

You must respond with ONLY a JSON object in one of these two formats:

If the article IS about the same case:
{"relevant": true, "confidence": "high|medium|low", "reason": "Brief explanation in English of why this article matches the case.", "summary": "संक्षिप्त सारांश (१-२ वाक्यमा नेपालीमा लेख्नुहोस्): यस समाचारले के भनेको छ — को संलग्न छ, के भयो, मुख्य तथ्यहरू। यो सार्वजनिक रूपमा देखाइने विवरण हो।", "event_type": "investigation|filing|hearing|verdict|appeal|other"}

If the article is NOT about the same case:
{"relevant": false, "reason": "Brief explanation in English of why this article does not match.", "summary": "A concise 1-3 sentence summary of what the news article reports."}

Event types for the "event_type" field:
- "investigation" — CIAA is investigating or completed investigation phase
- "filing" — CIAA filed the charge sheet at Special Court
- "hearing" — court hearing, proceedings, or bench session coverage
- "verdict" — Special Court verdict, decision, or conviction
- "appeal" — Supreme Court appeal or review
- "other" — relevant article that does not clearly fit the above categories

Rules:
- The article must reference the same corruption case, not just mention the same person in an unrelated context.
- Matching on case number alone is strong evidence of relevance.
- Matching on defendant name + corruption allegations is medium evidence.
- If the article is about a different corruption case involving the same person, it is NOT relevant.
- If the article is about the same person but not about corruption allegations, it is NOT relevant.
- The "summary" field should describe the article content itself in Nepali, not how the LLM matched it to the case.
- The "event_type" field is required when relevant=true. Never return an empty string. Use "other" if the article is relevant but does not clearly fit the above categories.
"""


class _TextExtractor(HTMLParser):
    """Extract visible text from HTML, skipping script/style tags."""

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self._skip = False
        self._skip_tags = {"script", "style", "noscript", "nav", "footer", "header"}

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self._skip_tags:
            self._skip = True

    def handle_endtag(self, tag):
        if tag.lower() in self._skip_tags:
            self._skip = False
        if tag.lower() in ("p", "br", "li", "div", "h1", "h2", "h3", "h4", "h5", "h6"):
            self.text_parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            text = data.strip()
            if text:
                self.text_parts.append(text)


def _fix_mojibake(text: str) -> str:
    """Repair UTF-8 text that was decoded as Latin-1 then re-encoded.

    The pattern `à¤...` is the hallmark of Devanagari script that got
    Latin-1-mangled somewhere in the HTTP → HTML → Python pipeline.
    """
    if not text:
        return text
    try:
        # Encoding as Latin-1 and decoding as UTF-8 reverses the double-encoding
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def _extract_text_from_html(html: str) -> str:
    """Extract visible text from HTML."""
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        logger.exception("HTML parse error in _extract_text_from_html")
    text = " ".join(parser.text_parts)
    text = re.sub(r"\s+", " ", text).strip()
    return _fix_mojibake(text)


def _extract_title_from_html(html: str) -> str:
    """Extract title from HTML <title> tag."""
    safe_html = _truncate_for_regex(html)
    match = re.search(r"<title[^>]*>([^<]*)</title>", safe_html, re.IGNORECASE)
    if match:
        title = re.sub(r"\s+", " ", match.group(1)).strip()
        return _fix_mojibake(title)
    return ""


def _search_duckduckgo(query: str, timeout: int = 15) -> list[dict]:
    """Search DuckDuckGo HTML and return list of result dicts.

    Retries with exponential backoff on 403/429 responses.
    Returns list of dicts with keys: title, url, snippet.
    """
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, headers=_HTTP_HEADERS, timeout=timeout)
            if resp.status_code in (403, 429):
                if attempt < max_attempts:
                    delay = 5 * 3 ** (attempt - 1)
                    logger.warning(
                        "DDG search attempt %d/%d: HTTP %d for '%s' — retrying in %ds",
                        attempt,
                        max_attempts,
                        resp.status_code,
                        query[:60],
                        delay,
                    )
                    time.sleep(delay)
                    continue
                logger.warning(
                    "DDG search failed after %d attempts for '%s'",
                    max_attempts,
                    query[:60],
                )
                return []
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning(
                "DuckDuckGo search failed for query '%s': %s", query[:60], exc
            )
            return []
        break  # success → exit loop

    html = _truncate_for_regex(resp.text)
    results = []

    link_pattern = re.compile(
        r'<a[^>]{0,200}class="result__a"[^>]{0,100}href="([^"]{1,500})"[^>]{0,50}>'
        r"([^<]{1,500})</a>",
        re.IGNORECASE,
    )
    snippet_pattern = re.compile(
        r'<a[^>]{0,200}class="result__snippet"[^>]{0,100}>([^<]{1,1000})</a>',
        re.IGNORECASE,
    )

    links = link_pattern.findall(html)
    snippets = snippet_pattern.findall(html)

    for i, (href, title_html) in enumerate(links):
        result_url = _extract_ddg_redirect(href)
        title_text = re.sub(r"<[^>]+>", "", title_html).strip()
        snippet_text = ""
        if i < len(snippets):
            snippet_text = re.sub(r"<[^>]+>", "", snippets[i]).strip()

        if result_url and title_text:
            results.append(
                {"title": title_text, "url": result_url, "snippet": snippet_text}
            )

    return results


def _extract_ddg_redirect(url: str) -> str:
    """Extract the real URL from DuckDuckGo redirect URL."""
    if "uddg=" in url:
        from urllib.parse import parse_qs, unquote

        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        uddg = params.get("uddg", [""])[0]
        if uddg:
            return unquote(uddg)
    return url


def _resolve_case_number(case: Case) -> Optional[str]:
    """Extract the first court case number from case.court_cases."""
    if not case.court_cases:
        return None
    for cc in case.court_cases:
        if isinstance(cc, str) and ":" in cc:
            return cc.split(":", 1)[1]
    return None


def _generate_query_variations(
    case: Case, llm_english_queries: Optional[list[str]] = None
) -> list[str]:
    """Generate search query variations for a CIAA case.

    Prioritizes accused name + location + corruption keywords over case numbers,
    which mostly surface court/admin pages rather than newsrooms.
    """
    case_number = _resolve_case_number(case)
    title = case.title or ""
    accused_names = _get_accused_names(case)

    event_queries: list[str] = []
    _append_event_targeted_queries(event_queries, case)

    english_queries: list[str] = []
    if llm_english_queries:
        english_queries.extend(llm_english_queries)
    else:
        _append_english_name_queries(english_queries, accused_names)

    general_queries = _build_name_based_queries(case)
    _append_title_keyword_query(general_queries, title)
    _append_accused_corruption_queries(general_queries, case)
    _append_location_queries(general_queries, case, title)

    deduped_english = _deduplicate_queries(english_queries, case_number, accused_names)
    deduped_events = _deduplicate_queries(event_queries, case_number, accused_names)
    deduped_general = _deduplicate_queries(general_queries, case_number, accused_names)

    reserved_english_slots = min(_QUERY_RESERVED_ENGLISH_SLOTS, len(deduped_english))
    reserved_event_slots = min(_QUERY_RESERVED_EVENT_SLOTS, len(deduped_events))
    general_slots = _QUERY_LIMIT - reserved_english_slots - reserved_event_slots

    combined = (
        deduped_english[:reserved_english_slots]
        + deduped_events[:reserved_event_slots]
        + deduped_general[:general_slots]
        + deduped_events[reserved_event_slots:]
        + deduped_english[reserved_english_slots:]
    )
    return _normalize_search_queries(
        _deduplicate_queries(combined, case_number, accused_names)
    )[:_QUERY_LIMIT]


_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")


_ROMANIZATION_REPLACEMENTS = (
    ("क्ष", "ksh"),
    ("त्र", "tr"),
    ("ज्ञ", "gy"),
    ("श्र", "shr"),
    ("अ", "a"),
    ("आ", "aa"),
    ("इ", "i"),
    ("ई", "ee"),
    ("उ", "u"),
    ("ऊ", "oo"),
    ("ए", "e"),
    ("ऐ", "ai"),
    ("ओ", "o"),
    ("औ", "au"),
    ("क", "k"),
    ("ख", "kh"),
    ("ग", "g"),
    ("घ", "gh"),
    ("ङ", "n"),
    ("च", "ch"),
    ("छ", "chh"),
    ("ज", "j"),
    ("झ", "jh"),
    ("ञ", "n"),
    ("ट", "t"),
    ("ठ", "th"),
    ("ड", "d"),
    ("ढ", "dh"),
    ("ण", "n"),
    ("त", "t"),
    ("थ", "th"),
    ("द", "d"),
    ("ध", "dh"),
    ("न", "n"),
    ("प", "p"),
    ("फ", "ph"),
    ("ब", "b"),
    ("भ", "bh"),
    ("म", "m"),
    ("य", "y"),
    ("र", "r"),
    ("ल", "l"),
    ("व", "w"),
    ("श", "sh"),
    ("ष", "sh"),
    ("स", "s"),
    ("ह", "h"),
    ("ा", "a"),
    ("ि", "i"),
    ("ी", "i"),
    ("ु", "u"),
    ("ू", "u"),
    ("े", "e"),
    ("ै", "ai"),
    ("ो", "o"),
    ("ौ", "au"),
    ("ं", "n"),
    ("ँ", "n"),
    ("ः", ""),
    ("्", ""),
)


def _romanize_devanagari(text: str) -> str:
    romanized = text
    for devanagari, roman in _ROMANIZATION_REPLACEMENTS:
        romanized = romanized.replace(devanagari, roman)
    romanized = re.sub(r"[^A-Za-z0-9\s\"-]", " ", romanized)
    return re.sub(r"\s+", " ", romanized).strip()


def _is_english_query(query: str) -> bool:
    return not _DEVANAGARI_RE.search(query) and bool(re.search(r"[A-Za-z]", query))


def _query_has_nepal_keyword(query: str) -> bool:
    return bool(re.search(r"(?:\bNepal\b|नेपाल)", query, flags=re.IGNORECASE))


def _with_nepal_keyword(query: str) -> str:
    if _query_has_nepal_keyword(query):
        return query
    return f"{query} {'नेपाल' if _DEVANAGARI_RE.search(query) else 'Nepal'}"


def _append_english_name_queries(queries: list[str], accused_names: list[str]) -> None:
    name = accused_names[0] if accused_names else ""
    roman_name = _romanize_devanagari(name)
    if roman_name and len(roman_name) >= 3:
        queries.append(f"{roman_name} CIAA Nepal corruption")
        queries.append(f"{roman_name} Nepal special court case")


def _normalize_search_queries(queries: list[str]) -> list[str]:
    normalized = [_with_nepal_keyword(query) for query in queries]
    english = [query for query in normalized if _is_english_query(query)]
    devanagari = [query for query in normalized if query not in english]
    return english[:4] + devanagari + english[4:]


def _append_title_keyword_query(queries: list[str], title: str) -> None:
    title_keywords = _extract_title_keywords(title)
    if title_keywords:
        queries.append(f"{title_keywords} भ्रष्टाचार")


def _append_accused_corruption_queries(queries: list[str], case: Case) -> None:
    key_allegations = case.key_allegations or []
    corruption_keywords = _extract_corruption_keywords(key_allegations)
    accused_names = _get_accused_names(case)

    for name in accused_names[:2]:
        name_clean = re.sub(r"\s+", " ", name).strip()
        if not name_clean or len(name_clean) < 3:
            continue
        for kw in corruption_keywords[:2]:
            queries.append(f"{name_clean} {kw} Nepal")


def _append_location_queries(queries: list[str], case: Case, title: str) -> None:
    accused_names = _get_accused_names(case)
    if not accused_names:
        return
    name_clean = re.sub(r"\s+", " ", accused_names[0]).strip()
    if not name_clean or len(name_clean) <= 3:
        return
    location = _extract_location_from_title(title)
    if location:
        queries.append(f"{name_clean} {location} भ्रष्टाचार")
        queries.append(f"{name_clean} {location} अख्तियार")


def _append_event_targeted_queries(queries: list[str], case: Case) -> None:
    """Generate event-targeted search queries from court case lifecycle data.

    Uses NGM hearing data when available; falls back to court_cases field
    to detect special/supreme court entries and case_start/end_dates.

    NGM integration is optional — failures are silent and fall back to
    field-based detection.
    """
    accused_names = _get_accused_names(case)
    if not accused_names:
        return
    name_clean = re.sub(r"\s+", " ", accused_names[0]).strip()
    if not name_clean or len(name_clean) < 3:
        return

    detected_events = _detect_case_events(case)
    for event_type in detected_events:
        templates = _EVENT_QUERY_TEMPLATES.get(event_type, [])
        for template in templates:
            queries.append(template.format(name=name_clean))


def _infer_event_type_from_reason(reason: str, fallback_event_type: str) -> str:
    """Extract a lifecycle event type from the LLM's reason text.

    When the structured event_type field is none/empty/unknown but the
    reason text clearly describes a specific event (e.g. "the article reports
    the CIAA filing a corruption charge sheet"), infer the correct type so
    event caps are applied correctly instead of silently falling through as
    "other".
    """
    reason_lower = reason.lower()

    # Start with the fallback that the LLM gave us (make "other" from
    # none/empty/unknown, but still try to improve it from reason).
    event = fallback_event_type if fallback_event_type in _LIFECYCLE_EVENT_TYPES else ""

    if not event:
        # Map reason keywords to event types, checked in priority order
        # so "appeal" doesn't match "supreme court verdict" (verdict wins
        # by position, which is wrong — appeal should lose to verdict).
        # Use most-specific-first ordering.
        if re.search(r"investigat(ion|e|ing)", reason_lower):
            event = _EVENT_INVESTIGATION
        elif re.search(
            r"charge.?sheet|filed|filing|filed a|charge.?sheet", reason_lower
        ):
            event = _EVENT_FILING
        elif re.search(r"hearing|proceeding|bench", reason_lower):
            event = _EVENT_HEARING
        elif re.search(
            r"verdict|convict|acquit|sentence|decision|found guilty", reason_lower
        ):
            event = _EVENT_VERDICT
        elif re.search(r"appeal|appellate|supreme court", reason_lower):
            event = _EVENT_APPEAL

    return event


def _detect_case_events(case: Case) -> list[str]:
    """Return all five lifecycle event types for broad coverage.

    The LLM filters out irrelevant results, so casting wider is safe.
    """
    return [
        _EVENT_FILING,
        _EVENT_INVESTIGATION,
        _EVENT_HEARING,
        _EVENT_VERDICT,
        _EVENT_APPEAL,
    ]


def _deduplicate_queries(
    queries: list[str], case_number: Optional[str], accused_names: list[str]
) -> list[str]:
    seen = set()
    deduped = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            deduped.append(q)

    if case_number:
        name_clean = (
            re.sub(r"\s+", " ", accused_names[0]).strip() if accused_names else ""
        )
        if name_clean and len(name_clean) > 3:
            deduped.append(f'"{case_number}" {name_clean}')

    return deduped


def _build_name_based_queries(case: Case) -> list[str]:
    """Build search queries from accused names and locations."""
    queries = []
    accused_names = _get_accused_names(case)
    title = case.title or ""

    location = _extract_location_from_title(title)
    org_name = _extract_org_name_from_title(title)

    for idx, name in enumerate(accused_names[:3]):
        name_clean = re.sub(r"\s+", " ", name).strip()
        if not name_clean or len(name_clean) < 3:
            continue
        if idx == 0:
            if location:
                queries.append(f'"{name_clean}" {location} भ्रष्टाचार')
                queries.append(f"{name_clean} {location} अख्तियार")
            queries.append(f"{name_clean} CIAA corruption")
        else:
            queries.append(f"{name_clean} CIAA corruption Nepal")

    if org_name:
        queries.append(f"{org_name} भ्रष्टाचार")
        queries.append(f"{org_name} अख्तियार")

    return queries


def _extract_org_name_from_title(title: str) -> str:
    """Extract organization name from case title for search queries."""
    if not title:
        return ""
    suffixes = (
        "सहकारी",
        "संस्था",
        "कम्पनी",
        "स्कुल",
        "कलेज",
        "अस्पताल",
        "बैंक",
        "विकास बैंक",
        "फाइनान्स",
        "जलस्रोत",
        "खानेपानी",
        "उपभोक्ता समिति",
        "विद्युत",
        "सिंचाइ",
        "निर्माण सेवा",
    )
    for suffix in sorted(suffixes, key=len, reverse=True):
        m = re.search(rf"(\S{{2,60}}\s*{re.escape(suffix)})", title)
        if m:
            return m.group(1).strip()
    return ""


def _extract_location_from_title(title: str) -> str:
    """Extract a location name from case title."""
    location_match = re.search(
        r"(?:कार्यालय|नगरपालिका|गाउँपालिका|जिल्ला)\s+(\S{1,50})",
        title,
    )
    if location_match:
        return location_match.group(1)
    loc_match2 = re.search(r"(\S{1,50})(?:को|का|मा)\s+(?:नापी|मालपोत|स्वास्थ्य)", title)
    if loc_match2:
        return loc_match2.group(1)
    return ""


def _extract_title_keywords(title: str) -> str:
    """Extract meaningful keywords from case title for search."""
    if not title:
        return ""
    parts = re.split(r"[,।\n]", title)
    if len(parts) > 1 and len(parts[0].strip()) > 10:
        return parts[0].strip()[:80]
    cleaned = re.sub(r"\b(?:मुद्दा|विरुद्ध|सम्बन्धी|सम्बन्धमा|मा\.?)\b", "", title)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:100]


def _get_accused_names(case: Case) -> list[str]:
    """Extract accused entity names from a case, preferring DB entity relationships."""
    names = []
    try:
        accused_rels = (
            CaseEntityRelationship.objects.filter(
                case=case, relationship_type=RelationshipType.ACCUSED
            )
            .select_related("entity")
            .order_by("created_at")
        )
        for rel in accused_rels:
            entity_name = rel.entity.display_name or rel.entity.nes_id or ""
            name_clean = entity_name.strip()
            if name_clean:
                names.append(name_clean)
    except Exception:
        logger.exception(
            "Failed to load accused entity relationships for case %s", case.case_id
        )
        close_old_connections()

    if names:
        return names[:5]

    if case.title:
        match = re.search(
            r"(?:विरुद्ध|vs\.?|versus)\s+(.{1,200})(?:\s+मुद्दा|\s+मा\.?\s|$)",
            case.title,
        )
        if match:
            rest = match.group(1).strip()
            if " र " in rest:
                names.extend(n.strip() for n in rest.split(" र "))
            elif "," in rest:
                names.extend(n.strip() for n in rest.split(","))
            else:
                names.append(rest)

    if not names and case.title:
        names.append(case.title[:80])

    return names[:5]


def _extract_corruption_keywords(key_allegations: list[str]) -> list[str]:
    """Extract corruption-related keywords from allegations for search queries."""
    corruption_terms = [
        "घुस",
        "रिश्वत",
        "भ्रष्टाचार",
        "अवैध सम्पत्ति",
        "हिनामिना",
        "पद दुरुपयोग",
        "किर्ते",
        "नक्कली",
        "बिगो",
        "अख्तियार",
        "विशेष अदालत",
        "bribery",
        "corruption",
        "illegal property",
        "embezzlement",
        "forgery",
        "abuse of authority",
    ]
    found = []
    text = " ".join(key_allegations).lower()
    for term in corruption_terms:
        if term.lower() in text:
            found.append(term)
    return found[:3]


def _fetch_article_content(url: str, timeout: int = 20) -> Optional[str]:
    """Fetch article HTML content from URL.

    Returns raw HTML string or None on failure.
    """
    try:
        resp = requests.get(
            url, headers=_FETCH_HEADERS, timeout=timeout, allow_redirects=True
        )
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "").lower()
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            return None
        return resp.text
    except requests.RequestException as exc:
        logger.debug("Failed to fetch %s: %s", url, exc)
        return None


def _detect_truncated_response(data: dict) -> bool:
    """Check if the LLM response was truncated due to max_tokens limit."""
    choices = data.get("choices", [])
    if not choices:
        return False
    choice = choices[0] if isinstance(choices[0], dict) else {}
    finish_reason = choice.get("finish_reason", "") if isinstance(choice, dict) else ""
    return finish_reason == "length"


def _resolve_llm_content(data: dict) -> Optional[str]:
    """Safely extract content from LLM response, supporting OpenAI and alternate shapes."""
    choices = data.get("choices", [])
    if not choices:
        return None
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        text_parts = [
            part.get("text")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        return "\n".join(text_parts) if text_parts else None
    if isinstance(content, str):
        return content
    content = choice.get("text") if isinstance(choice, dict) else None
    if content is not None:
        return content
    content = message.get("text") if isinstance(message, dict) else None
    if content is not None:
        return content
    content = choice.get("content") if isinstance(choice, dict) else None
    return content


def _fallback_parse_relevance(raw_text: str) -> Optional[dict]:
    """Try to extract relevance from raw response when JSON parse fails."""
    match = re.search(r'"relevant"\s*:\s*(true|false)', raw_text)
    if match:
        is_relevant = match.group(1) == "true"
        conf_match = re.search(r'"confidence"\s*:\s*"([^"]+)"', raw_text)
        reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', raw_text)
        summary_match = re.search(r'"summary"\s*:\s*"([^"]*)"', raw_text)
        return {
            "relevant": is_relevant,
            "confidence": conf_match.group(1) if conf_match else "low",
            "reason": (
                reason_match.group(1)
                if reason_match
                else "fallback: extracted from raw response"
            ),
            "summary": summary_match.group(1) if summary_match else "",
        }
    return None


def _safe_parse_http_json(raw_text: str) -> dict:
    """Parse an HTTP JSON body tolerantly.

    Uses raw_decode to handle concatenated streaming chunks where the proxy
    appended multiple JSON objects instead of streaming a single payload.
    """
    text = raw_text.strip()
    try:
        decoder = json.JSONDecoder()
        obj, _end = decoder.raw_decode(text)
        return obj
    except json.JSONDecodeError:
        pass
    raise json.JSONDecodeError("raw_decode failed for HTTP JSON body", text[:500], 0)


def _call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str,
    base_url: str,
    api_key: str,
    timeout: int = 60,
    max_retries: int = 3,
) -> Optional[dict]:
    """Call LLM API with exponential backoff retry and return parsed JSON response."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 1200,
        "stream": False,
    }

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = 2**attempt
                logger.warning(
                    "LLM call attempt %d/%d failed: %s — retrying in %ds",
                    attempt,
                    max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
                continue
            logger.warning(
                "LLM call failed after %d attempts: %s", max_retries, last_exc
            )
            return None

        try:
            data = _safe_parse_http_json(resp.text)
        except json.JSONDecodeError as exc:
            logger.warning("LLM response JSON decode failed: %s", exc)
            return None

        if _detect_truncated_response(data):
            logger.warning(
                "LLM response truncated (finish_reason=length) — "
                "response may be incomplete; consider increasing max_tokens or reducing prompt size"
            )

        content = _resolve_llm_content(data)
        if content is None:
            logger.debug(
                "LLM response missing recognized content field: %s",
                json.dumps(data)[:500],
            )
            return _fallback_parse_relevance(json.dumps(data))
        result = _parse_llm_json(content)
        if result is None:
            logger.debug("LLM JSON parse failed; raw content: %s", content[:500])
            return _fallback_parse_relevance(content)
        return result

    return None


def _parse_llm_json(text: str) -> Optional[dict]:
    """Extract and parse the first complete JSON object from LLM response text.

    Uses json.JSONDecoder.raw_decode() to find the boundary of the first valid
    JSON object, then ignores any trailing text (explanations, markdown fences,
    or a second JSON object the model may have appended).
    """
    text = text.strip()
    json_start = text.find("{")
    if json_start == -1:
        return None
    try:
        decoder = json.JSONDecoder()
        obj, _end = decoder.raw_decode(text, json_start)
        return obj
    except json.JSONDecodeError:
        # Strip markdown code fences and retry
        cleaned = re.sub(r"^```(?:json)?[ \t]*", "", text, flags=re.MULTILINE)
        if "```" in cleaned:
            fence_idx = cleaned.rindex("```")
            cleaned = cleaned[:fence_idx].rstrip()
        try:
            decoder = json.JSONDecoder()
            obj, _end = decoder.raw_decode(cleaned, cleaned.find("{"))
            return obj
        except json.JSONDecodeError:
            return None


class NewsEnricher:
    """Service for enriching CIAA cases with related news articles.

    Pipeline per case:
    1. Generate search queries from case metadata
    2. Search for candidate news articles
    3. Deduplicate candidate URLs
    4. Fetch article content
    5. Verify relevance with LLM
    6. Extract article metadata and images
    7. Store as DocumentSource with MEDIA_NEWS type
    8. Link to Case.evidence
    """

    def __init__(
        self,
        llm_model: str = _DEFAULT_LLM_MODEL,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        max_articles_per_case: int = 5,
        search_delay: float = 1.5,
        fetch_delay: float = 0.5,
        verbose: bool = False,
    ):
        self.llm_model = llm_model or _DEFAULT_LLM_MODEL
        self.llm_base_url = llm_base_url or os.environ.get(
            "JAWAFDEHI_LLM_PROXY_URL", _DEFAULT_LLM_BASE_URL
        )
        self.llm_api_key = llm_api_key
        self.max_articles_per_case = max_articles_per_case
        self.search_delay = search_delay
        self.fetch_delay = fetch_delay
        self.verbose = verbose
        self._existing_url_map: dict[str, str] = {}
        self._llm_configured = (
            bool(llm_api_key)
            or bool(os.environ.get("JAWAFDEHI_LLM_API_KEY"))
            or bool(os.environ.get("ANTHROPIC_API_KEY"))
        )

    def _resolve_api_key(self, cli_key: Optional[str]) -> Optional[str]:
        """Resolve LLM API key."""
        if cli_key:
            return cli_key
        return os.environ.get("JAWAFDEHI_LLM_API_KEY") or os.environ.get(
            "ANTHROPIC_API_KEY"
        )

    def enrich_case(
        self,
        case: Case,
        dry_run: bool = False,
        force: bool = False,
        case_num: int = 0,
        total_cases: int = 0,
    ) -> dict:
        """Enrich a single case with news articles.

        Returns stats dict with status and counters.
        """
        api_key = self._resolve_api_key(self.llm_api_key)
        stats = self._validate_prerequisites(api_key, dry_run)
        if stats is not None:
            return stats

        case_number = _resolve_case_number(case)
        self._log_case_progress(case, case_number, case_num, total_cases)

        case_linked_urls = self._get_case_linked_urls(case)
        _, self._existing_url_map = self._get_existing_url_metadata()

        if self._is_already_saturated(case, force):
            return self._make_stats("skipped", "already_saturated")

        press_release_text = self._get_press_release_content(case)
        if press_release_text:
            logger.info(
                "  INFO: Press release context: %d chars", len(press_release_text)
            )
        else:
            logger.warning(
                "  WARNING: no press release text — LLM verification will lack official case context"
            )

        llm_english_queries = self._generate_english_queries_via_llm(
            case, press_release_text
        )

        queries = _generate_query_variations(
            case, llm_english_queries=llm_english_queries
        )
        if not queries:
            logger.info("  No search queries generated")
            stats = self._make_stats("skipped")
            stats["reason"] = "no_queries"
            return stats

        accepted, stats = self._perform_enrichment(
            case, queries, api_key, case_linked_urls, press_release_text, force
        )

        if not accepted and stats.get("already_linked", 0) == 0:
            stats["status"] = "no_articles"
            return stats

        stats["new_sources"] = self._handle_enrichment_results(case, accepted, dry_run)
        return stats

    def _validate_prerequisites(self, api_key, dry_run):
        if not dry_run and not api_key:
            return {
                "status": "skipped",
                "reason": "no_llm_key",
                "searched": 0,
                "fetched": 0,
                "accepted": 0,
                "rejected": 0,
                "errors": 0,
                "already_linked": 0,
                "new_sources": 0,
            }
        if dry_run and not api_key:
            logger.warning(
                "No LLM API key configured — article relevance verification disabled. "
                "Set JAWAFDEHI_LLM_API_KEY, ANTHROPIC_API_KEY, or use --llm-api-key."
            )
        return None

    def _is_already_saturated(self, case, force):
        current_media_news_count = self._count_media_news_evidence(case)
        if current_media_news_count >= self.max_articles_per_case and not force:
            logger.info(
                "  Already has %d MEDIA_NEWS evidence entries (max=%d) — skipping",
                current_media_news_count,
                self.max_articles_per_case,
            )
            return True
        return False

    @staticmethod
    def _make_stats(status="processed", reason=""):
        stats = {
            "status": status,
            "searched": 0,
            "fetched": 0,
            "accepted": 0,
            "rejected": 0,
            "errors": 0,
            "already_linked": 0,
            "new_sources": 0,
        }
        if reason:
            stats["reason"] = reason
        return stats

    def _perform_enrichment(
        self, case, queries, api_key, case_linked_urls, press_release_text, force
    ):
        stats = self._make_stats()
        current_media_news_count = self._count_media_news_evidence(case)

        all_candidates = self._search_candidates(queries, stats)
        new_candidates, already_linked = self._filter_case_candidates(
            all_candidates, case_linked_urls, force
        )

        if already_linked > 0:
            logger.info("  %d URLs already linked as evidence", already_linked)
            stats["already_linked"] = already_linked

        if not new_candidates and already_linked > 0 and not force:
            stats["status"] = "skipped"
            stats["reason"] = "all_already_linked"
            return [], stats

        remaining_slots = self.max_articles_per_case - current_media_news_count
        if remaining_slots <= 0 and not force:
            stats["status"] = "skipped"
            stats["reason"] = "max_articles_reached"
            return [], stats

        accepted = self._fetch_and_verify_candidates(
            new_candidates,
            case,
            api_key,
            stats,
            press_release_text,
            max_to_accept=remaining_slots,
        )

        if not accepted and stats.get("already_linked", 0) == 0:
            accepted = self._retry_with_fallback_queries(
                case=case,
                api_key=api_key,
                stats=stats,
                case_linked_urls=case_linked_urls,
                remaining_slots=remaining_slots,
                press_release_text=press_release_text,
                force=force,
            )

        return accepted, stats

    def _search_candidates(self, queries: list[str], stats: dict) -> list[dict]:
        """Execute search queries sequentially and collect deduplicated results."""
        all_candidates = []
        seen_urls = set()

        for index, query in enumerate(queries):
            stats["searched"] += 1
            try:
                results = _search_duckduckgo(query)
            except Exception as exc:
                logger.warning("  Search error for '%s': %s", query[:60], exc)
                stats["errors"] += 1
                results = []

            new_count = 0
            for r in results:
                url = r["url"]
                if url not in seen_urls:
                    seen_urls.add(url)
                    all_candidates.append(r)
                    new_count += 1
            logger.info(
                "  query '%s' → %d results (%d new)",
                query[:70],
                len(results),
                new_count,
            )

            if index < len(queries) - 1 and self.search_delay > 0:
                time.sleep(self.search_delay)

        logger.info(
            "  Found %d candidate URLs from %d queries",
            len(all_candidates),
            len(queries),
        )
        return all_candidates

    _RETRY_MAX = 3

    def _generate_fallback_queries(self, case: Case, attempt: int) -> list[str]:
        """Generate progressively simpler fallback search queries.

        Each retry attempt uses a different strategy to maximize the odds of
        finding articles that the primary query variations missed.
        """
        if attempt == 0:
            return self._fallback_queries_attempt_0(case)
        if attempt == 1:
            return self._fallback_queries_attempt_1(case)
        return self._fallback_queries_attempt_2(case)

    def _fallback_queries_attempt_0(self, case: Case) -> list[str]:
        accused_names = _get_accused_names(case)
        title = case.title or ""
        queries = []
        for name in accused_names[:2]:
            name_clean = re.sub(r"\s+", " ", name).strip()
            if name_clean and len(name_clean) >= 3:
                queries.append(f'"{name_clean}" Nepal')
        if title and len(title) > 10:
            queries.append(title[:100])
        return queries[:5]

    def _fallback_queries_attempt_1(self, case: Case) -> list[str]:
        accused_names = _get_accused_names(case)
        queries = []
        for name in accused_names[:2]:
            name_clean = re.sub(r"\s+", " ", name).strip()
            if name_clean and len(name_clean) >= 3:
                queries.append(f"{name_clean} corruption Nepal")
                queries.append(f"{name_clean} भ्रष्टाचार")
        return queries[:5]

    def _fallback_queries_attempt_2(self, case: Case) -> list[str]:
        accused_names = _get_accused_names(case)
        title = case.title or ""
        title_keywords = _extract_title_keywords(title)
        if title_keywords:
            return [f"{title_keywords} Nepal"]
        if accused_names:
            name_clean = re.sub(r"\s+", " ", accused_names[0]).strip()
            if name_clean and len(name_clean) >= 3:
                return [f"{name_clean} Nepal"]
        return ["CIAA Nepal corruption"]

    def _retry_with_fallback_queries(
        self,
        case: Case,
        api_key: Optional[str],
        stats: dict,
        case_linked_urls: set[str],
        remaining_slots: int,
        press_release_text: Optional[str],
        force: bool,
    ) -> list[dict]:
        """Retry enrichment with fallback query strategies when initial search yields 0 articles.

        Retries up to _RETRY_MAX times, each with different query structures.
        Stops early if any retry finds ≥1 accepted article.
        """
        for attempt in range(self._RETRY_MAX):
            fallback_queries = self._generate_fallback_queries(case, attempt)
            logger.info(
                "  Retry %d/%d: fallback search with %d queries",
                attempt + 1,
                self._RETRY_MAX,
                len(fallback_queries),
            )

            retry_candidates = self._search_candidates(fallback_queries, stats)

            new_candidates, retry_linked = self._filter_case_candidates(
                retry_candidates, case_linked_urls, force
            )
            if retry_linked > 0:
                stats["already_linked"] += retry_linked

            if not new_candidates:
                logger.info(
                    "  Retry %d/%d: no new candidates found",
                    attempt + 1,
                    self._RETRY_MAX,
                )
                continue

            accepted = self._fetch_and_verify_candidates(
                new_candidates,
                case,
                api_key,
                stats,
                press_release_text,
                max_to_accept=remaining_slots,
            )

            if accepted:
                logger.info(
                    "  Retry %d/%d: found %d accepted article(s)",
                    attempt + 1,
                    self._RETRY_MAX,
                    len(accepted),
                )
                return accepted

            logger.info(
                "  Retry %d/%d: 0 accepted articles",
                attempt + 1,
                self._RETRY_MAX,
            )

        logger.info(
            "  All %d retries exhausted — marking as no_articles", self._RETRY_MAX
        )
        return []

    # Candidates are fetched and verified in batches so we can stop early
    # once all event types are covered, avoiding unnecessary network + LLM work.
    _CANDIDATE_BATCH_SIZE = 12

    def _fetch_and_verify_candidates(
        self,
        candidates: list[dict],
        case: Case,
        api_key: Optional[str],
        stats: dict,
        press_release_text: Optional[str] = None,
        max_to_accept: Optional[int] = None,
    ) -> list[dict]:
        """Fetch and verify candidates in batches. Stops early when all event types covered."""
        limit = (
            max_to_accept if max_to_accept is not None else self.max_articles_per_case
        )

        accepted: list[dict] = []
        accepted_event_types: set[str] = set()
        event_type_counts = self._get_existing_event_type_counts(case)
        all_deferred: list[dict] = []

        def _would_exceed_event_cap(event: str) -> bool:
            return (
                event
                and event_type_counts.get(event, 0) >= _MAX_ARTICLES_PER_EVENT_TYPE
            )

        def _uncovered_events() -> list[str]:
            return [
                et
                for et in _ALL_EVENT_TYPES
                if et not in accepted_event_types and not _would_exceed_event_cap(et)
            ]

        for batch_start in range(0, len(candidates), self._CANDIDATE_BATCH_SIZE):
            if len(accepted) >= limit:
                break
            if not _uncovered_events():
                break

            batch = candidates[batch_start : batch_start + self._CANDIDATE_BATCH_SIZE]
            fetched = [
                r
                for r in self._fetch_candidates_parallel(batch, stats)
                if r is not None
            ]

            if not fetched:
                continue

            batch_def = self._verify_candidates_parallel(
                fetched,
                case,
                api_key,
                stats,
                press_release_text,
                accepted,
                accepted_event_types,
                event_type_counts,
                _would_exceed_event_cap,
            )
            all_deferred.extend(batch_def)

            batch_num = batch_start // self._CANDIDATE_BATCH_SIZE + 1
            total_batches = (
                len(candidates) + self._CANDIDATE_BATCH_SIZE - 1
            ) // self._CANDIDATE_BATCH_SIZE
            logger.info(
                "  Batch %d/%d: %d accepted (%d event types), %d deferred",
                batch_num,
                total_batches,
                len(accepted),
                len(accepted_event_types),
                len(batch_def),
            )

        self._fill_deferred_slots(
            all_deferred,
            limit,
            accepted,
            accepted_event_types,
            event_type_counts,
            stats,
            _would_exceed_event_cap,
        )

        return accepted

    def _verify_candidates_parallel(
        self,
        fetched: list[dict],
        case: Case,
        api_key: Optional[str],
        stats: dict,
        press_release_text: Optional[str],
        accepted: list[dict],
        accepted_event_types: set[str],
        event_type_counts: dict[str, int],
        _would_exceed_event_cap,
    ) -> list[dict]:
        """Verify fetched candidates with LLM in parallel. Returns deferred list."""
        deferred: list[dict] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_result = {}
            for result in fetched:
                future = executor.submit(
                    self._verify_candidate,
                    result,
                    case,
                    api_key,
                    stats,
                    press_release_text,
                )
                future_to_result[future] = result

            for future in concurrent.futures.as_completed(future_to_result):
                try:
                    verified = future.result()
                except Exception as exc:
                    logger.warning("  LLM verify error: %s", exc)
                    stats["errors"] += 1
                    continue
                if verified:
                    event = verified.get("event_type", "")
                    if (
                        event
                        and event not in accepted_event_types
                        and not _would_exceed_event_cap(event)
                    ):
                        accepted.append(verified)
                        accepted_event_types.add(event)
                        event_type_counts[event] = event_type_counts.get(event, 0) + 1
                        stats["accepted"] += 1
                        logger.info(
                            "  Accepted: %s (event: %s, confidence: %s)",
                            verified["url"][:80],
                            event,
                            verified["confidence"],
                        )
                    else:
                        deferred.append(verified)
                        logger.debug(
                            "  Deferred (duplicate event %s): %s",
                            event or "unknown",
                            verified["url"][:80],
                        )

        return deferred

    def _fill_deferred_slots(
        self,
        deferred: list[dict],
        limit: int,
        accepted: list[dict],
        accepted_event_types: set[str],
        event_type_counts: dict[str, int],
        stats: dict,
        _would_exceed_event_cap,
    ) -> None:
        """Fill remaining accepted slots from deferred candidates."""
        remaining = limit - len(accepted)
        used_indices: set[int] = set()

        for i, verified in enumerate(deferred):
            if remaining <= 0:
                break
            event = verified.get("event_type", "")
            if (
                event
                and event not in accepted_event_types
                and not _would_exceed_event_cap(event)
            ):
                accepted.append(verified)
                accepted_event_types.add(event)
                event_type_counts[event] = event_type_counts.get(event, 0) + 1
                stats["accepted"] += 1
                remaining -= 1
                used_indices.add(i)
                logger.info(
                    "  Accepted (deferred): %s (event: %s, confidence: %s)",
                    verified["url"][:80],
                    event,
                    verified["confidence"],
                )

        for i, verified in enumerate(deferred):
            if remaining <= 0:
                break
            if i in used_indices:
                continue
            event = verified.get("event_type", "")
            if event and _would_exceed_event_cap(event):
                continue
            accepted.append(verified)
            event_type_counts[event] = event_type_counts.get(event, 0) + 1
            stats["accepted"] += 1
            remaining -= 1
            logger.info(
                "  Accepted (same event): %s (event: %s, confidence: %s)",
                verified["url"][:80],
                event,
                verified["confidence"],
            )

    def _fetch_candidates_parallel(
        self,
        candidates: list[dict],
        stats: dict,
    ) -> list[Optional[dict]]:
        """Fetch candidate articles in parallel using ThreadPoolExecutor."""

        def _fetch_one(candidate: dict) -> Optional[dict]:
            url = candidate["url"]
            if _is_official_press_release(url):
                logger.debug("  skipped: official CIAA press release — %s", url[:80])
                return None
            blocklist_reason = _is_url_blocklisted(url)
            if blocklist_reason:
                logger.debug("  skipped: %s — %s", blocklist_reason, url[:80])
                return None
            try:
                html = _fetch_article_content(url)
                time.sleep(self.fetch_delay)
                if not html:
                    logger.debug("  Failed to fetch content: %s", url)
                    return None
                article_text = _extract_text_from_html(html)
                if len(article_text) < 100:
                    logger.debug(
                        "  Insufficient content from %s (%d chars)",
                        url,
                        len(article_text),
                    )
                    return None
                article_title = _extract_title_from_html(html) or candidate.get(
                    "title", ""
                )
                article_date = _extract_publication_date(html)
                stats["fetched"] += 1
                return {
                    "candidate": candidate,
                    "article_text": article_text,
                    "article_title": article_title,
                    "article_date": article_date,
                }
            except Exception as exc:
                exc_type = type(exc).__name__
                logger.warning("  fetch error: %s — %s: %s", exc_type, url[:80], exc)
                stats["errors"] += 1
                return None

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_candidate = {
                executor.submit(_fetch_one, c): c for c in candidates
            }
            for future in concurrent.futures.as_completed(future_to_candidate):
                try:
                    result = future.result()
                except Exception as exc:
                    logger.warning("  Unhandled fetch error: %s", exc)
                    stats["errors"] += 1
                    result = None
                results.append(result)
        return results

    def _verify_candidate(
        self,
        fetch_result: dict,
        case: Case,
        api_key: Optional[str],
        stats: dict,
        press_release_text: Optional[str] = None,
    ) -> Optional[dict]:
        """Run LLM verification on a single fetched article. Returns accepted dict or None."""
        candidate = fetch_result["candidate"]
        url = candidate["url"]
        article_text = fetch_result["article_text"]
        article_title = fetch_result["article_title"]

        if api_key and self.llm_base_url:
            is_relevant, confidence, reason, summary, event_type = (
                self._verify_article_relevance(
                    case=case,
                    article_title=article_title,
                    article_url=url,
                    article_excerpt=article_text[:1500],
                    api_key=api_key,
                    press_release_text=press_release_text,
                )
            )
        else:
            is_relevant = False
            confidence = "none"
            reason = "LLM not configured"
            summary = ""
            event_type = ""

        if event_type is None or str(event_type).strip().lower() in {
            "",
            "none",
            "unknown",
        }:
            inferred = _infer_event_type_from_reason(reason, event_type or "")
            event_type = inferred if inferred else "other"

        if not is_relevant:
            stats["rejected"] += 1
            if self.verbose:
                logger.info("  rejected: %s — %s", url[:80], reason)
            else:
                logger.debug("  rejected: %s — %s", url[:80], reason)
            return None

        return {
            "title": article_title,
            "url": url,
            "publication_date": fetch_result.get("article_date"),
            "confidence": confidence,
            "reason": reason,
            "summary": summary,
            "event_type": event_type or "",
        }

    @staticmethod
    def _extract_urls_from_source(source: DocumentSource) -> list[str]:
        """Extract individual URL strings from a DocumentSource's url field."""
        if isinstance(source.url, list):
            return [u for u in source.url if isinstance(u, str)]
        return []

    @staticmethod
    def _extract_source_ids_from_evidence(evidence: list) -> set[str]:
        """Extract source_id values from case evidence entries."""
        if not evidence:
            return set()
        return {
            entry["source_id"]
            for entry in evidence
            if isinstance(entry, dict) and entry.get("source_id")
        }

    def _get_existing_event_type_counts(self, case: Case) -> dict[str, int]:
        """Count existing MEDIA_NEWS evidence entries by event_type."""
        evidence = case.evidence or []
        source_ids = self._extract_source_ids_from_evidence(evidence)
        if not source_ids:
            return {}

        try:
            media_news_source_ids = set(
                DocumentSource.objects.filter(
                    source_id__in=source_ids,
                    source_type=SourceType.MEDIA_NEWS,
                ).values_list("source_id", flat=True)
            )
        except Exception:
            logger.exception("Failed to load existing MEDIA_NEWS event types")
            close_old_connections()
            return {}

        counts: dict[str, int] = {}
        for entry in evidence:
            if not isinstance(entry, dict):
                continue
            if entry.get("source_id") not in media_news_source_ids:
                continue
            event_type = entry.get("event_type")
            if not event_type:
                continue
            counts[event_type] = counts.get(event_type, 0) + 1
        return counts

    def _index_source_url(
        self, urls: set[str], url_map: dict[str, str], source
    ) -> None:
        """Add source URL strings to the dedup set and URL→source_id map."""
        if not isinstance(source.url, list):
            return
        for u in source.url:
            if isinstance(u, str):
                urls.add(u)
                if u not in url_map:
                    url_map[u] = source.source_id

    def _get_existing_url_metadata(self) -> tuple[set[str], dict[str, str]]:
        """Get existing article URLs and URL→source_id mapping in one pass."""
        urls = set()
        url_map = {}
        try:
            for source in DocumentSource.objects.filter(
                source_type=SourceType.MEDIA_NEWS, is_deleted=False
            ).only("url", "source_id"):
                self._index_source_url(urls, url_map, source)
        except Exception:
            logger.exception("Failed to load existing URL metadata")
            close_old_connections()
        return urls, url_map

    def _count_media_news_evidence(self, case: Case) -> int:
        """Count evidence entries that reference MEDIA_NEWS sources (including soft-deleted).

        Uses the evidence JSON list as the ground-truth count so stale entries
        (soft-deleted or type-changed sources) still count toward the limit.
        """
        source_ids = self._extract_source_ids_from_evidence(case.evidence)
        if not source_ids:
            return 0
        try:
            return DocumentSource.objects.filter(
                source_id__in=source_ids,
                source_type=SourceType.MEDIA_NEWS,
            ).count()
        except Exception:
            logger.exception("Failed to count MEDIA_NEWS evidence entries")
            close_old_connections()
            return 0

    def _get_case_linked_urls(self, case: Case) -> set[str]:
        """Get set of URLs already linked to this specific case via evidence."""
        linked_urls = set()
        source_ids = self._extract_source_ids_from_evidence(case.evidence)
        if not source_ids:
            return linked_urls
        try:
            for source in DocumentSource.objects.filter(
                source_id__in=source_ids,
                source_type=SourceType.MEDIA_NEWS,
                is_deleted=False,
            ).only("url"):
                linked_urls.update(self._extract_urls_from_source(source))
        except Exception:
            logger.exception("Failed to fetch linked URLs for case")
            close_old_connections()
        return linked_urls

    def _log_case_progress(
        self, case: Case, case_number: Optional[str], case_num: int, total_cases: int
    ) -> None:
        """Log enrichment progress for a case."""
        if case_num and total_cases:
            cn_str = f" (#{case_number})" if case_number else ""
            logger.info(
                "[%d/%d] Processing %s%s — %s",
                case_num,
                total_cases,
                case.case_id,
                cn_str,
                (case.title or "")[:70],
            )
        else:
            logger.info("Processing %s...", case.case_id)

    def _filter_case_candidates(
        self,
        all_candidates: list[dict],
        case_linked_urls: set[str],
        force: bool,
    ) -> tuple[list[dict], int]:
        """Split candidates into new vs already-linked.

        Only URLs already in this case's evidence are counted as already-linked.
        URLs that exist globally but aren't linked to this case are processed
        so the existing DocumentSource can be attached to this case's evidence.

        When force=True, all candidates are treated as new (already-linked
        count is still reported but does not prevent processing).
        """
        already_linked = sum(1 for c in all_candidates if c["url"] in case_linked_urls)
        if force:
            return list(all_candidates), already_linked

        new_candidates = []
        for c in all_candidates:
            url = c["url"]
            if url in case_linked_urls:
                continue
            else:
                new_candidates.append(c)
        return new_candidates, already_linked

    def _handle_enrichment_results(
        self, case: Case, accepted: list[dict], dry_run: bool
    ) -> int:
        """Save articles or log dry-run, then log final article state for the case.

        Returns new_sources count.
        """
        pre_existing = self._count_media_news_evidence(case)

        new_sources = 0
        if not dry_run and accepted:
            new_sources = self._save_articles(case, accepted)
        elif dry_run and accepted:
            logger.info("  [DRY RUN] Would save %d article(s)", len(accepted))
            for a in accepted:
                logger.info("    - %s", a["url"][:80])
            new_sources = len(accepted)

        self._log_case_article_summary(case, pre_existing, new_sources, accepted)
        return new_sources

    def _log_case_article_summary(
        self,
        case: Case,
        pre_existing: int,
        new_sources: int,
        accepted: Optional[list[dict]] = None,
    ) -> None:
        """Log the final news article state for a case after enrichment."""
        source_ids = self._extract_source_ids_from_evidence(case.evidence)
        if not source_ids:
            logger.info("  Final: 0 MEDIA_NEWS articles linked")
            return

        try:
            sources = list(
                DocumentSource.objects.filter(
                    source_id__in=source_ids,
                    source_type=SourceType.MEDIA_NEWS,
                ).only("source_id", "title", "url")
            )
        except Exception:
            logger.exception("Failed to fetch article state for logging")
            close_old_connections()
            return

        total = len(sources)
        limit_status = (
            "AT LIMIT" if total >= self.max_articles_per_case else "BELOW LIMIT"
        )

        logger.info("  --- Article Summary ---")
        logger.info("  Existing articles:   %d", pre_existing)
        logger.info("  Added in this run:   %d", new_sources)
        logger.info("  Final total:         %d [%s]", total, limit_status)
        for s in sources:
            url_str = (
                self._extract_urls_from_source(s)[0]
                if self._extract_urls_from_source(s)
                else "?"
            )
            logger.info("    - %s | %s", s.title or "Untitled", url_str)

        if accepted:
            event_counts: dict[str, int] = {}
            event_examples: dict[str, str] = {}
            for a in accepted:
                et = a.get("event_type", "")
                if et:
                    event_counts[et] = event_counts.get(et, 0) + 1
                    if et not in event_examples:
                        event_examples[et] = a.get("title", "")[:60]
            if event_counts:
                parts = []
                for et in _ALL_EVENT_TYPES:
                    if et in event_counts:
                        parts.append(f"{et}={event_counts[et]}")
                logger.info("  Event coverage (accepted in run): %s", ", ".join(parts))

    def _generate_english_queries_via_llm(
        self, case: Case, press_release_text: Optional[str]
    ) -> list[str]:
        """Use LLM to generate correctly romanized English search queries.

        The character-map romanizer produces broken transliterations
        (वहादुर → wahadur instead of Bahadur). LLM knows correct spelling.

        Returns up to 5 English queries (one per event type) with correctly
        romanized names. On any failure, returns empty list so caller falls
        back to romanization path.
        """
        api_key = self._resolve_api_key(self.llm_api_key)
        if not api_key or not self.llm_base_url:
            return []

        accused_names = _get_accused_names(case)
        name_list = ", ".join(accused_names[:3]) if accused_names else "Unknown"
        case_context = f"Case Title: {case.title or 'Unknown'}\nAccused: {name_list}"

        user_prompt = (
            "Generate 5 English search queries to find Nepali news articles "
            "about this CIAA corruption case. One query per event type: "
            "investigation, chargesheet filing, court hearing, verdict, appeal. "
            "Use correct English romanization of Nepali names (e.g. Bahadur not wahadur, Bisht not wisht). "
            "Include the word Nepal in each query. "
            'Respond with ONLY: {"queries": ["q1", "q2", "q3", "q4", "q5"]}\n\n'
            f"{case_context}"
        )

        try:
            result = _call_llm(
                system_prompt="You are a Nepal-focused news search assistant. Output only clean search queries.",
                user_prompt=user_prompt,
                model=self.llm_model,
                base_url=self.llm_base_url,
                api_key=api_key,
                timeout=30,
                max_retries=1,
            )
        except Exception:
            logger.debug("LLM English query generation failed", exc_info=True)
            return []

        if result is None:
            return []

        queries = result.get("queries") if isinstance(result, dict) else None
        if not isinstance(queries, list):
            return []

        english_queries = [
            q
            for q in queries
            if isinstance(q, str) and _is_english_query(q) and len(q) > 10
        ][:5]

        if english_queries:
            logger.info(
                "  LLM generated %d English queries (e.g. %s)",
                len(english_queries),
                english_queries[0][:80],
            )
        else:
            logger.debug("  LLM English queries empty or unparseable")

        return english_queries

    def _verify_article_relevance(
        self,
        case: Case,
        article_title: str,
        article_url: str,
        article_excerpt: str,
        api_key: str,
        press_release_text: Optional[str] = None,
    ) -> tuple[bool, str, str, str, str]:
        """Use LLM to verify if an article is about the same case.

        Returns (is_relevant, confidence, reason, summary).
        """
        case_number = _resolve_case_number(case)

        case_context = f"""Case Title: {case.title or 'Unknown'}
Court Case Number: {case_number or 'Unknown'}
Short Description: {case.short_description or 'Not provided'}
Key Allegations: {', '.join(case.key_allegations[:5]) if case.key_allegations else 'None'}"""

        if press_release_text:
            case_context += f"""

Press Release Text (official CIAA document):
{press_release_text[:1200]}"""
        else:
            case_context += """

No official press release text available."""

        user_prompt = f"""Determine if this news article is about the same corruption case.

CASE CONTEXT:
{case_context}

ARTICLE:
Title: {article_title}
URL: {article_url}
Excerpt: {article_excerpt}"""

        logger.debug(
            "LLM verify context for %s: case=%s case#=%s press_release_chars=%d article_title=%s",
            article_url[:80],
            case.case_id,
            case_number or "none",
            len(press_release_text) if press_release_text else 0,
            article_title[:80],
        )

        result = _call_llm(
            system_prompt=_VERIFY_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=self.llm_model,
            base_url=self.llm_base_url,
            api_key=api_key,
            timeout=120,
        )

        if result is None:
            return False, "error", "LLM response could not be parsed", "", ""

        relevant = result.get("relevant", False)
        confidence = result.get("confidence", "medium")
        reason = result.get("reason", "")
        summary = result.get("summary", "")
        event_type = result.get("event_type", "")

        logger.debug(
            "LLM verdict for %s: relevant=%s confidence=%s event=%s reason=%s",
            article_url[:80],
            relevant,
            confidence,
            event_type or "none",
            reason,
        )

        return relevant, confidence, reason, summary, event_type

    def _get_press_release_content(self, case: Case) -> Optional[str]:
        """Extract press release text for a CIAA case from evidence-linked sources.

        Strategy:
        1. Look for DocumentSource records linked via evidence with press release
           content in the description field.
        2. For sources with ciaa.gov.np URLs, try downloading the corresponding
           NGM-stored PDF via ngm-store.jawafdehi.org and converting via likhit.
        3. Collect content from all matching sources.
        """
        source_ids = self._extract_source_ids_from_evidence(case.evidence)
        if not source_ids:
            logger.debug("  No valid source_ids in evidence")
            return None

        sources = list(
            DocumentSource.objects.filter(
                source_id__in=source_ids, is_deleted=False
            ).only("source_id", "description", "title", "url")
        )
        if not sources:
            logger.debug(
                "  No matching DocumentSource records found (%d IDs)", len(source_ids)
            )
            return None

        source_by_id = {s.source_id: s for s in sources}
        press_release_parts = self._collect_press_release_parts(
            source_ids, source_by_id
        )

        if not press_release_parts:
            logger.debug(
                "  No press release text extracted from %d matching source(s)",
                sum(
                    1
                    for sid in source_ids
                    if source_by_id.get(sid)
                    and self._is_press_release_source(source_by_id[sid])
                ),
            )
            return None

        combined = "\n\n".join(press_release_parts)
        logger.debug(
            "  Press release content from %d source(s): %d total chars",
            len(press_release_parts),
            len(combined),
        )
        return combined

    def _collect_press_release_parts(
        self, source_ids: list[str], source_by_id: dict[str, DocumentSource]
    ) -> list[str]:
        """Collect press release text from descriptions and downloadable URLs."""
        parts = []
        for sid in source_ids:
            source = source_by_id.get(sid)
            if source is None or not self._is_press_release_source(source):
                continue
            self._extract_text_from_source(source, parts)
        return parts

    def _extract_pdf_from_source(self, source: DocumentSource) -> Optional[str]:
        """Try to extract text from PDF URLs in an allowed-host source.
        Returns the converted markdown content or None.
        """
        if not isinstance(source.url, list):
            return None
        for url in source.url:
            if not self._is_pdf_url(url):
                continue
            parsed = urlparse(url)
            if not parsed.hostname or parsed.hostname not in _ALLOWED_HOSTS:
                continue
            content = self._convert_to_markdown(url)
            if content and len(content) > 200:
                return content
        return None

    def _extract_text_from_source(
        self, source: DocumentSource, parts: list[str]
    ) -> None:
        """Extract press release text from a source's description or .pdf URLs only.

        Ignores .doc/.docx files and web/HTML pages — only .pdf files from
        allowed hosts are downloaded and converted.
        """
        description = (source.description or "").strip()
        if len(description) > 200:
            parts.append(description)
            logger.debug(
                "  Press release text from source %s: %d chars",
                source.source_id,
                len(description),
            )
            return

        content = self._extract_pdf_from_source(source)
        if content:
            parts.append(content)
            logger.debug(
                "  Press release text from URL: %d chars",
                len(content),
            )

    @staticmethod
    def _is_pdf_url(url: str) -> bool:
        """Return True if the URL points to a .pdf file."""
        parsed = urlparse(url)
        path = parsed.path.lower()
        return path.endswith(".pdf")

    def _is_press_release_source(self, source: DocumentSource) -> bool:
        """Check if a DocumentSource is a CIAA press release.

        Excludes court order documents (judgments, verdicts, orders).
        """
        title_lower = (source.title or "").lower()
        if self._is_court_order_title(title_lower):
            return False

        if "press release" in title_lower or "ciaa" in title_lower:
            return True

        if isinstance(source.url, list):
            for url in source.url:
                parsed = urlparse(url)
                if parsed.hostname and parsed.hostname in _ALLOWED_HOSTS:
                    return True
        return False

    @staticmethod
    def _is_court_order_title(title_lower: str) -> bool:
        """Check if a title suggests a court order document rather than a press release."""
        court_keywords = (
            "court order",
            "judgment",
            "verdict",
            "फैसला",
            "आदेश",
            "निर्णय",
        )
        return any(kw in title_lower for kw in court_keywords)

    def _convert_to_markdown(self, url: str) -> Optional[str]:
        """Download file from URL and convert to markdown using likhit/markitdown.

        Pipeline: URL download -> temp file -> likhit/markitdown -> markdown.
        Returns None when conversion fails or produces insufficient content.
        """
        try:
            response = requests.get(url, timeout=120, stream=True)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("  Failed to download %s: %s", url, exc)
            return None

        final_hostname = urlparse(response.url).hostname
        if final_hostname not in _ALLOWED_HOSTS:
            logger.warning("  Redirected to untrusted host: %s", response.url)
            return None

        content_type = response.headers.get("content-type", "").lower()

        text_result = self._handle_text_response(response, content_type)
        if text_result is not None:
            return text_result

        return self._convert_binary_response(response, content_type, url)

    def _handle_text_response(self, response, content_type: str) -> Optional[str]:
        """Handle plain text or JSON responses directly."""
        if "text/plain" in content_type or "application/json" in content_type:
            response.encoding = "utf-8"
            text = response.text
            if len(text) > 200:
                return text
            return None
        return None

    @staticmethod
    def _determine_suffix(content_type: str) -> str:
        """Determine file suffix from content-type header."""
        if "pdf" in content_type:
            return ".pdf"
        if "html" in content_type:
            return ".html"
        if any(kw in content_type for kw in ("document", "word", "docx", "msword")):
            return ".docx"
        return ""

    def _convert_binary_response(
        self, response, content_type: str, url: str
    ) -> Optional[str]:
        """Save response to temp file and convert via markitdown."""
        import importlib.util

        suffix = self._determine_suffix(content_type)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = tmp.name
                for chunk in response.iter_content(chunk_size=8192):
                    tmp.write(chunk)

            if importlib.util.find_spec("likhit"):
                import likhit  # noqa: F401

            from markitdown import MarkItDown

            md = MarkItDown(enable_plugins=True)
            result = md.convert(tmp_path)

            if (
                result
                and result.text_content
                and len(result.text_content.strip()) > 200
            ):
                return result.text_content.strip()

            logger.warning(
                "  Markitdown conversion produced insufficient content for %s", url
            )
            return None
        except Exception as exc:
            logger.warning("  Markitdown conversion failed for %s: %s", url, exc)
            return None
        finally:
            if tmp_path:
                try:
                    Path(tmp_path).unlink()
                except OSError:
                    pass

    def _save_articles(self, case: Case, articles: list[dict]) -> int:
        """Save accepted articles as DocumentSource and link to case evidence.

        Re-checks the MEDIA_NEWS evidence count right before saving to prevent
        over-storage from stale/raced limit checks. Caps accepted articles
        to remaining slots even if verification returned more.

        Returns number of new sources created.
        """
        new_count = 0
        evidence = list(case.evidence) if case.evidence else []
        existing_source_ids = self._extract_source_ids_from_evidence(evidence)
        url_to_source = self._existing_url_map

        # Pre-save re-check: count evidence entries (including soft-deleted)
        # so we don't overshoot the limit even if the early check was stale.
        current_count = DocumentSource.objects.filter(
            source_id__in=existing_source_ids,
            source_type=SourceType.MEDIA_NEWS,
        ).count()
        capped_slots = max(0, self.max_articles_per_case - current_count)
        articles_to_save = articles[:capped_slots]

        if capped_slots < len(articles):
            logger.info(
                "  Pre-save limit check: capping %d accepted to %d slots "
                "(current=%d, max=%d)",
                len(articles),
                capped_slots,
                current_count,
                self.max_articles_per_case,
            )

        with transaction.atomic():
            for article in articles_to_save:
                url = article["url"]
                existing_source_id = url_to_source.get(url)

                if existing_source_id:
                    if existing_source_id not in existing_source_ids:
                        evidence.append(
                            self._build_evidence_entry(existing_source_id, article)
                        )
                        existing_source_ids.add(existing_source_id)
                    continue

                source = self._create_document_source(
                    article, fallback_date=case.case_start_date
                )
                url_to_source[url] = source.source_id
                new_count += 1

                if source.source_id not in existing_source_ids:
                    evidence.append(
                        self._build_evidence_entry(source.source_id, article)
                    )
                    existing_source_ids.add(source.source_id)

            case.evidence = self._sort_media_news_evidence(evidence)
            case.save(update_fields=["evidence", "updated_at"])

        return new_count

    def _sort_media_news_evidence(self, evidence: list[dict]) -> list[dict]:
        """Keep non-news evidence first, then sort MEDIA_NEWS by lifecycle event_type."""
        source_ids = self._extract_source_ids_from_evidence(evidence)
        if not source_ids:
            return evidence

        try:
            media_news_source_ids = set(
                DocumentSource.objects.filter(
                    source_id__in=source_ids,
                    source_type=SourceType.MEDIA_NEWS,
                ).values_list("source_id", flat=True)
            )
        except Exception:
            logger.exception("Failed to sort MEDIA_NEWS evidence entries")
            close_old_connections()
            return evidence

        non_media_news = []
        media_news = []
        for index, entry in enumerate(evidence):
            if (
                isinstance(entry, dict)
                and entry.get("source_id") in media_news_source_ids
            ):
                media_news.append((index, entry))
            else:
                non_media_news.append(entry)

        media_news.sort(
            key=lambda indexed: (
                _EVENT_LIFECYCLE_ORDER.get(
                    indexed[1].get("event_type"), _EVENT_LIFECYCLE_ORDER[_EVENT_OTHER]
                ),
                indexed[0],
            )
        )
        return non_media_news + [entry for _, entry in media_news]

    def _create_document_source(
        self, article: dict, fallback_date: Optional[date] = None
    ) -> DocumentSource:
        """Create a DocumentSource record for an accepted news article."""
        description = _fix_mojibake(self._build_source_description(article))
        title = (
            _fix_mojibake(article["title"])
            if article.get("title")
            else "Untitled News Article"
        )
        pub_date = article.get("publication_date") or fallback_date or date.today()

        source = DocumentSource(
            title=title,
            description=description,
            source_type=SourceType.MEDIA_NEWS,
            url=[article["url"]],
            publication_date=pub_date,
        )
        source.save()
        return source

    def _build_source_description(self, article: dict) -> str:
        """Build a public-facing article description from the LLM summary.

        Falls back to outlet + date when summary is missing. Omits missing
        fields — never writes placeholders like "unknown date".
        """
        summary = article.get("summary")
        if summary:
            return summary

        outlet = _guess_outlet(article.get("url", ""))
        pub_date = article.get("publication_date")
        if pub_date:
            return f"{outlet}ले यस मुद्दासम्बन्धी समाचार प्रकाशित गरेको ({pub_date.isoformat()})।"
        return f"{outlet}ले यस मुद्दासम्बन्धी समाचार प्रकाशित गरेको।"

    def _build_evidence_entry(self, source_id: str, article: dict) -> dict:
        """Build case evidence entry for a MEDIA_NEWS source."""
        entry = {
            "source_id": source_id,
            "description": self._build_evidence_description(article),
        }
        event_type = article.get("event_type")
        if event_type:
            entry["event_type"] = event_type
        return entry

    def _build_evidence_description(self, article: dict) -> str:
        """Build evidence entry description in Nepali."""
        outlet = _guess_outlet(article.get("url", ""))
        pub_date = article.get("publication_date")

        date_str = ""
        if pub_date:
            date_str = f" ({pub_date.isoformat()})"

        return _fix_mojibake(
            f"{outlet}{date_str} ले यस मुद्दासम्बन्धी समाचार प्रकाशित गरेको।"
        )


def _guess_outlet(url: str) -> str:
    """Guess news outlet name from URL."""
    try:
        hostname = urlparse(url).hostname or ""
        hostname = re.sub(r"^www\d*\.", "", hostname)
        parts = hostname.split(".")
        if len(parts) >= 2:
            return parts[-2].title()
        return hostname
    except Exception:
        return "Unknown"


def _parse_date_string(date_str: str) -> Optional[date]:
    """Try to parse a date string using common formats. Returns date or None."""
    date_str = date_str[:19]
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(date_str[: len(fmt.replace("%Z", ""))], fmt)
            return dt.date()
        except ValueError:
            continue
    return None


def _extract_publication_date(html: str) -> Optional[date]:
    """Extract publication date from HTML meta tags or content."""
    safe_html = _truncate_for_regex(html)
    patterns = [
        r'<meta[^>]+?property="article:published_time"[^>]+?content="([^"]+)"',
        r'<meta[^>]+?name="[^"]*date[^"]*"[^>]+?content="([^"]+)"',
        r'<meta[^>]+?itemprop="datePublished"[^>]+?content="([^"]+)"',
        r'"datePublished"\s*:\s*"([^"]+)"',
    ]

    for pattern in patterns:
        match = re.search(pattern, safe_html, re.IGNORECASE)
        if match:
            result = _parse_date_string(match.group(1))
            if result is not None:
                return result

    nepali_date_pattern = re.compile(
        r"(?:प्रकाशित|मिति)[:\s]*(\d{4})[-/](\d{1,2})[-/](\d{1,2})"
    )
    match = nepali_date_pattern.search(safe_html)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            pass

    return None


def enrich_cases_batch(
    enricher: NewsEnricher,
    cases,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """Enrich multiple cases and return aggregate stats."""
    stats = {
        "total": 0,
        "processed": 0,
        "skipped": 0,
        "searched": 0,
        "fetched": 0,
        "accepted": 0,
        "rejected": 0,
        "errors": 0,
        "already_linked": 0,
        "new_sources": 0,
        "cases_with_articles": 0,
        "cases_no_articles": 0,
    }

    total = 0
    cases_list = []
    for c in cases:
        cases_list.append(c)
        total += 1

    for idx, case in enumerate(cases_list, 1):
        close_old_connections()
        stats["total"] += 1
        try:
            result = enricher.enrich_case(
                case,
                dry_run=dry_run,
                force=force,
                case_num=idx,
                total_cases=total,
            )
            if result["status"] in ("skipped", "no_articles"):
                stats["skipped"] += 1
            else:
                stats["processed"] += 1

            stats["searched"] += result.get("searched", 0)
            stats["fetched"] += result.get("fetched", 0)
            stats["accepted"] += result.get("accepted", 0)
            stats["rejected"] += result.get("rejected", 0)
            stats["errors"] += result.get("errors", 0)
            stats["already_linked"] += result.get("already_linked", 0)
            stats["new_sources"] += result.get("new_sources", 0)
            if result.get("accepted", 0) > 0 or result.get("already_linked", 0) > 0:
                stats["cases_with_articles"] += 1
            elif result.get("status") == "no_articles":
                stats["cases_no_articles"] += 1
        except Exception as exc:
            stats["errors"] += 1
            logger.exception("Failed to process %s: %s", case.case_id, exc)

    return stats
