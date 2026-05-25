"""Service for rule-based and LLM-based tag classification of CIAA cases."""

import ipaddress
import json
import logging
import os
import re
import socket
import tempfile
import urllib.request
from urllib.parse import urlparse

from cases.models import Case, DocumentSource, DocumentSourceUpload, SourceType

logger = logging.getLogger(__name__)

SECTOR_TAGS = [
    "Local Government",
    "Health",
    "Education",
    "Infrastructure",
    "Land Management",
    "Finance",
    "Agriculture",
    "Energy",
    "Water Supply",
    "Transportation",
    "Telecommunications",
    "Forestry",
    "Tourism",
    "Revenue",
    "Banking",
    "IT",
    "Housing",
    "Construction",
    "Procurement",
    "Public Works",
    "Social Security",
    "Defense",
    "Foreign Affairs",
    "Home Affairs",
    "Law and Justice",
]

CORRUPTION_TYPE_TAGS = [
    "Bribery",
    "Illegal Property Acquisition",
    "Procurement Irregularities",
    "Public Office Abuse",
    "Embezzlement",
    "Forged Documents",
    "Revenue Leakage",
    "Nepotism",
    "Witness Tampering",
    "Bid Rigging",
    "Tax Evasion",
    "Money Laundering",
    "Assets Beyond Known Income",
    "Conflict of Interest",
    "Kickbacks",
]

REGION_TAGS = [
    "Province 1",
    "Madhesh",
    "Bagmati",
    "Gandaki",
    "Lumbini",
    "Karnali",
    "Sudurpashchim",
    "Kathmandu Valley",
    "Kathmandu",
    "Lalitpur",
    "Bhaktapur",
]

AMOUNT_TIER_TAGS = []  # Dynamic — generated from bigo field, not a fixed vocabulary

CONTEXT_TAGS = [
    "CIAA",
    "Special Court",
    "Supreme Court",
    "Corruption",
]

SECTOR_KEYWORDS = {
    "Local Government": [
        "municipality",
        "नगरपालिका",
        "local government",
        "स्थानीय",
        "vdc",
        "ward office",
        "वडा",
        "गाउँपालिका",
        "rural municipality",
        "metropolitan",
        "महानगर",
        "उपमहानगर",
        "sub-metropolitan",
    ],
    "Health": [
        "health",
        "स्वास्थ्य",
        "hospital",
        "अस्पताल",
        "medical",
        "चिकित्सा",
        "medicine",
        "औषधि",
        "pharmacy",
        "doctor",
    ],
    "Education": [
        "education",
        "शिक्षा",
        "school",
        "विद्यालय",
        "college",
        "क्याम्पस",
        "university",
        "विश्वविद्यालय",
        "teacher",
        "शिक्षक",
        "campus",
        "student",
        "professor",
        "प्राध्यापक",
    ],
    "Infrastructure": [
        "infrastructure",
        "पूर्वाधार",
        "road",
        "सडक",
        "bridge",
        "पुल",
        "airport",
        "विमानस्थल",
        "highway",
        "राजमार्ग",
    ],
    "Land Management": [
        "land",
        "जग्गा",
        "जमिन",
        "lalita niwas",
        "ललिता निवास",
        "land grab",
        "कित्ता",
        "plot",
        "ropani",
        "रोपनी",
        "land revenue",
        "मालपोत",
        "land registration",
        "survey",
        "नापी",
        "baluwatar",
        "बालुवाटार",
    ],
    "Finance": [
        "finance",
        "वित्त",
        "budget",
        "बजेट",
        "treasury",
        "कोष",
        "financial",
        "financial management",
        "fiscal",
        "audit",
        "लेखापरीक्षण",
    ],
    "Agriculture": [
        "agriculture",
        "कृषि",
        "farming",
        "खेती",
        "irrigation",
        "सिंचाइ",
        "fertilizer",
        "मल",
        "livestock",
        "पशुपालन",
    ],
    "Energy": [
        "energy",
        "ऊर्जा",
        "electricity",
        "विद्युत",
        "hydropower",
        "जलविद्युत",
        "सौर्य",
        "solar",
        "petroleum",
        "तेल",
    ],
    "Water Supply": [
        "water supply",
        "खानेपानी",
        "melamchi",
        "मेलम्ची",
        "drinking water",
        "sewerage",
        "ढल",
        "sanitation",
        "सरसफाइ",
        "water resource",
    ],
    "Transportation": [
        "transport",
        "यातायात",
        "vehicle",
        "सवारी",
        "railway",
        "रेल",
        "truck",
        "ट्रक",
        "airline",
    ],
    "Telecommunications": [
        "telecom",
        "दूरसञ्चार",
        "phone",
        "फोन",
        "internet",
        "इन्टरनेट",
        "ntc",
        "ncell",
        "frequency",
        "spectrum",
    ],
    "Forestry": [
        "forest",
        "वन",
        "timber",
        "काठ",
        "wildlife",
        "वन्यजन्तु",
        "national park",
        "निकुञ्ज",
        "deforestation",
        "logging",
    ],
    "Tourism": [
        "tourism",
        "पर्यटन",
        "hotel",
        "होटल",
        "casino",
        "क्यासिनो",
        "travel",
        "mountaineering",
        "trekking",
    ],
    "Revenue": [
        "revenue",
        "राजस्व",
        "tax",
        "कर",
        "customs",
        "भन्सार",
        "excise",
        "अन्तःशुल्क",
        "value added tax",
        "मूल्य अभिवृद्धि कर",
    ],
    "Banking": [
        "bank",
        "बैंक",
        "cooperative",
        "सहकारी",
        "loan",
        "ऋण",
        "credit",
        "कर्जा",
        "deposit",
        "निक्षेप",
        "microfinance",
        "लघुवित्त",
        "savings",
        "बचत",
    ],
    "IT": [
        "information technology",
        "सूचना प्रविधि",
        "software",
        "computer",
        "कम्प्युटर",
        "digital",
        "डिजिटल",
        "database",
        "server",
    ],
    "Housing": [
        "housing",
        "आवास",
        "apartment",
        "अपार्टमेन्ट",
        "building",
        "भवन",
        "colony",
        "real estate",
        "घरजग्गा",
        "settlement",
    ],
    "Construction": [
        "construction",
        "निर्माण",
        "building",
        "भवन",
        "contractor",
        "ठेकेदार",
        "civil works",
        "structure",
        "bridge construction",
    ],
    "Procurement": [
        "procurement",
        "खरिद",
        "tender",
        "टेन्डर",
        "supply",
        "आपूर्ति",
        "contract",
        "सम्झौता",
        "purchase",
        "bid",
    ],
    "Public Works": [
        "public works",
        "सार्वजनिक निर्माण",
        "public building",
        "government building",
    ],
}

CORRUPTION_TYPE_KEYWORDS = {
    "Bribery": [
        "bribe",
        "bribery",
        "घुस",
        "रिश्वत",
        "pay off",
        "illegal payment",
        "गैरकानूनी भुक्तानी",
        "undue advantage",
    ],
    "Illegal Property Acquisition": [
        "illegal property",
        "अवैध सम्पत्ति",
        "property acquisition",
        "disproportionate assets",
        "source unknown",
        "स्रोत नखुलेको",
        "illegal wealth",
        "अवैध धन",
        "property amassed",
        "benami",
        "बेनामी",
        "land grab",
        "जग्गा कब्जा",
    ],
    "Procurement Irregularities": [
        "procurement irregular",
        "खरिद अनियमितता",
        "tender manipulation",
        "contract violation",
        "supply fraud",
        "fake bill",
        "नक्कली बिल",
        "quality compromise",
        "गुणस्तरहीन",
        "over invoicing",
    ],
    "Public Office Abuse": [
        "abuse of authority",
        "abuse of office",
        "पद दुरुपयोग",
        "अख्तियार दुरुपयोग",
        "misuse of power",
        "misuse of position",
        "power abuse",
        "authority misuse",
        "official misconduct",
        "dereliction of duty",
        "कर्तव्य पालनामा लापरवाही",
    ],
    "Embezzlement": [
        "embezzl",
        "हिनामिना",
        "misappropriat",
        "अपचलन",
        "siphon",
        "fund diversion",
        "रकम अपचलन",
        "defalcation",
        "theft",
        "चोरी",
        "fraudulent transfer",
        "financial irregularity",
    ],
    "Forged Documents": [
        "forged",
        "forgery",
        "किर्ते",
        "fake document",
        "नक्कली",
        "false document",
        "fabricated",
        "बनावटी",
        "counterfeit",
        "tampered document",
        "fake certificate",
        "नक्कली प्रमाणपत्र",
        "forged signature",
    ],
    "Revenue Leakage": [
        "revenue leak",
        "राजस्व चुहावट",
        "tax avoidance",
        "customs fraud",
        "under invoicing",
        "smuggling",
        "तस्करी",
        "duty evasion",
    ],
    "Nepotism": [
        "nepotism",
        "crony",
        "nepot",
        "भाई-भतिजा",
        "कृपा",
        "favouritism",
        "favoritism",
        "relative appointment",
        "आफन्त",
        "nephew",
    ],
    "Witness Tampering": [
        "witness tamper",
        "witness intimidat",
        "साक्षी",
        "गवाह",
        "witness influenc",
        "evidence tamper",
        "प्रमाण",
        "threaten witness",
    ],
    "Bid Rigging": [
        "bid rig",
        "मिलेमतो",
        "collusion",
        "cartel",
        "price fixing",
        "rigged tender",
        "सेटिङ",
    ],
    "Tax Evasion": [
        "tax evasion",
        "tax fraud",
        "कर छली",
        "tax dodge",
        "false tax",
        "undeclared income",
        "hidden income",
    ],
    "Money Laundering": [
        "money launder",
        "सम्पत्ति शुद्धीकरण",
        "black money",
        "कालो धन",
        "hawala",
        "हुन्डी",
        "illicit finance",
        "shell company",
        "offshore",
        "round tripping",
    ],
    "Assets Beyond Known Income": [
        "assets beyond",
        "income source",
        "आयस्रोत",
        "wealth beyond",
        "property disproportionate",
        "सम्पत्ति विवरण",
        "lifestyle audit",
    ],
    "Conflict of Interest": [
        "conflict of interest",
        "स्वार्थ बाझिनु",
        "vested interest",
        "personal interest",
        "निजी स्वार्थ",
        "self-dealing",
    ],
    "Kickbacks": [
        "kickback",
        "commission",
        "कमिशन",
        "percentage cut",
        "दलाली",
        "brokerage",
        "middleman fee",
        "बिचौलिया",
    ],
}

REGION_KEYWORDS = {
    "Province 1": [
        "province 1",
        "कोशी",
        "koshi",
        "birtamod",
        "bhadrapur",
        "dharan",
        "धरान",
        "biratnagar",
        "विराटनगर",
        "illam",
        "इलाम",
        "jhapa",
        "झापा",
        "sunsari",
        "सुनसरी",
        "morang",
        "मोरङ",
        "sankhuwasabha",
        "taplejung",
        "terhathum",
        "bhojpur",
        "भोजपुर",
        "dhankuta",
        "धनकुटा",
        "khotang",
        "खोटाङ",
        "okhaldhunga",
        "solukhumbu",
        "udayapur",
        "उदयपुर",
    ],
    "Madhesh": [
        "madhesh",
        "मधेश",
        "janakpur",
        "जनकपुर",
        "birgunj",
        "वीरगन्ज",
        "saptari",
        "सप्तरी",
        "siraha",
        "सिराहा",
        "dhanusa",
        "धनुषा",
        "mahottari",
        "महोत्तरी",
        "sarlahi",
        "सर्लाही",
        "rautahat",
        "रौतहट",
        "bara",
        "बारा",
        "parsa",
        "पर्सा",
        "rajbiraj",
        "राजविराज",
    ],
    "Bagmati": [
        "bagmati",
        "बागमती",
        "hetauda",
        "हेटौंडा",
        "chitwan",
        "चितवन",
        "makwanpur",
        "मकवानपुर",
        "dhading",
        "धादिङ",
        "nuwakot",
        "नुवाकोट",
        "rasuwa",
        "रसुवा",
        "sindhupalchok",
        "सिन्धुपाल्चोक",
        "dolakha",
        "दोलखा",
        "kavre",
        "काभ्रे",
        "ramechhap",
        "रामेछाप",
        "sindhuli",
        "सिन्धुली",
    ],
    "Gandaki": [
        "gandaki",
        "गण्डकी",
        "pokhara",
        "पोखरा",
        "kaski",
        "कास्की",
        "lamjung",
        "लमजुङ",
        "tanahun",
        "तनहुँ",
        "gorakha",
        "गोरखा",
        "manang",
        "मनाङ",
        "mustang",
        "मुस्ताङ",
        "myagdi",
        "म्याग्दी",
        "parbat",
        "पर्वत",
        "syanja",
        "स्याङ्जा",
        "nawalparasi east",
        "baglung",
        "बागलुङ",
    ],
    "Lumbini": [
        "lumbini",
        "लुम्बिनी",
        "butwal",
        "बुटवल",
        "bhairahawa",
        "rupandehi",
        "रुपन्देही",
        "kapilvastu",
        "कपिलवस्तु",
        "nawalparasi west",
        "palpa",
        "पाल्पा",
        "arghakhanchi",
        "gulmi",
        "गुल्मी",
        "dang",
        "दाङ",
        "pyuthan",
        "प्युठान",
        "rolpa",
        "रोल्पा",
        "banke",
        "बाँके",
        "bardiya",
        "बर्दिया",
    ],
    "Karnali": [
        "karnali",
        "कर्णाली",
        "birendranagar",
        "surkhet",
        "सुर्खेत",
        "dailekh",
        "दैलेख",
        "jajarkot",
        "जाजरकोट",
        "rukum west",
        "salyan",
        "सल्यान",
        "kalikot",
        "कालिकोट",
        "jumla",
        "जुम्ला",
        "humla",
        "हुम्ला",
        "dolpa",
        "डोल्पा",
        "mugu",
        "मुगु",
    ],
    "Sudurpashchim": [
        "sudurpashchim",
        "सुदूरपश्चिम",
        "dhangadhi",
        "धनगढी",
        "kailali",
        "कैलाली",
        "kanchanpur",
        "कञ्चनपुर",
        "doti",
        "डोटी",
        "achham",
        "अछाम",
        "bajhang",
        "बझाङ",
        "bajura",
        "बाजुरा",
        "baitadi",
        "बैतडी",
        "darchula",
        "दार्चुला",
        "dadeldhura",
        "डडेल्धुरा",
    ],
    "Kathmandu": [
        "kathmandu",
        "काठमाडौं",
        "ktm",
        "kathmandu metropolitan",
        "kathmandu city",
        "kathmandu district",
    ],
    "Lalitpur": [
        "lalitpur",
        "ललितपुर",
        "patan",
        "पाटन",
        "godawari",
        "jawalakhel",
        "pulchowk",
    ],
    "Bhaktapur": [
        "bhaktapur",
        "भक्तपुर",
        "bhaktapur municipality",
    ],
    "Kathmandu Valley": [
        "kathmandu valley",
        "काठमाडौं उपत्यका",
    ],
}


def _collect_case_text(case: Case) -> str:
    """Build a searchable text blob from a case for keyword matching."""
    parts = [case.title or ""]
    if case.key_allegations:
        parts.extend(case.key_allegations)
    if case.court_cases:
        for cc in case.court_cases:
            if isinstance(cc, str):
                parts.append(cc)
    if case.description:
        parts.append(case.description)
    return " ".join(parts).lower()


def _collect_evidence_text(case: Case) -> str:  # noqa
    """Build a text blob from evidence entries, using actual file content when available.

    Priority:
    1. DocumentSource uploaded_file content (converted via MarkItDown)
    2. DocumentSourceUpload file content (converted via MarkItDown)
    3. DocumentSource title + description (metadata)
    4. Evidence entry description

    Filters to high-value source types (press releases, court orders) that
    contain the richest information for CIAA cases.
    """
    HIGH_VALUE_SOURCE_TYPES = (
        SourceType.LEGAL_PROCEDURAL,
        SourceType.LEGAL_COURT_ORDER,
    )

    parts = []
    if not case.evidence:
        logger.info("  No evidence entries found")
        return ""

    source_ids = []
    for entry in case.evidence:
        if isinstance(entry, dict):
            desc = entry.get("description", "")
            if desc:
                parts.append(desc)
            sid = entry.get("source_id", "")
            if sid:
                source_ids.append(sid)

    if not source_ids:
        logger.info("  No source IDs in evidence entries")
        return " ".join(parts)

    try:
        all_sources = DocumentSource.objects.filter(source_id__in=source_ids)
        high_value = all_sources.filter(source_type__in=HIGH_VALUE_SOURCE_TYPES)
        other = all_sources.exclude(source_type__in=HIGH_VALUE_SOURCE_TYPES)
    except Exception as e:
        logger.warning(f"  Failed to fetch DocumentSource records: {e}")
        return " ".join(parts)

    all_count = len(source_ids)
    hv_count = high_value.count()
    logger.info(
        f"  Found {hv_count}/{all_count} high-value sources (press releases, court orders)"
    )

    sources = list(high_value) + list(other)

    source_ids_for_uploads = [s.id for s in sources]
    pre_fetched_uploads = {}
    if source_ids_for_uploads:
        try:
            all_uploads = DocumentSourceUpload.objects.filter(
                source_id__in=source_ids_for_uploads
            )
            for upload in all_uploads:
                pre_fetched_uploads.setdefault(upload.source_id, []).append(upload)
        except Exception as e:
            logger.debug(f"Failed to prefetch DocumentSourceUpload: {e}")

    source_count = 0
    for src in sources:
        file_text = _convert_source_file(src, pre_fetched_uploads.get(src.id, []))
        if file_text:
            parts.append(file_text)
            source_count += 1
        else:
            if src.title:
                parts.append(src.title)
            if src.description:
                parts.append(src.description)

    result = " ".join(parts)
    logger.info(f"  Extracted {len(result)} chars from {source_count} source documents")
    return result


try:
    from markitdown import MarkItDown

    _MD_CONVERTER = MarkItDown(enable_plugins=False)
except ImportError:
    _MD_CONVERTER = None


_DOCUMENT_EXTENSIONS = frozenset({".pdf", ".doc", ".docx", ".jpg", ".jpeg", ".png"})


def _convert_source_file(src: DocumentSource, pre_fetched_uploads=None) -> str:  # noqa
    """Convert a DocumentSource's files to text using MarkItDown.

    Checks sources in order:
    1. Remote URLs in src.url (download .pdf/.doc/.docx, convert, clean up)
    2. Local uploaded_file on src
    3. Local DocumentSourceUpload files

    When pre_fetched_uploads is provided (list of DocumentSourceUpload),
    those are used directly instead of issuing a per-source DB query.

    Returns file content as markdown text, or empty string if conversion fails.
    """
    if _MD_CONVERTER is None:
        return ""

    url_result = _convert_urls(src)
    if url_result:
        return url_result

    files_to_try = []
    if src.uploaded_file and hasattr(src.uploaded_file, "path"):
        files_to_try.append(src.uploaded_file.path)

    if pre_fetched_uploads is not None:
        for upload in pre_fetched_uploads:
            if upload.file and hasattr(upload.file, "path"):
                files_to_try.append(upload.file.path)
    else:
        try:
            uploads = DocumentSourceUpload.objects.filter(source_id=src.id)
            for upload in uploads:
                if upload.file and hasattr(upload.file, "path"):
                    files_to_try.append(upload.file.path)
        except Exception as e:
            logger.debug(
                f"Failed to fetch DocumentSourceUpload for src.id={src.id}: {e}"
            )

    for filepath in files_to_try:
        try:
            result = _MD_CONVERTER.convert(filepath)
            if result and result.text_content:
                return result.text_content[:1200]
        except Exception as e:
            logger.debug(f"MarkItDown conversion failed for {filepath}: {e}")
            continue

    return ""


_MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024
_REQUEST_TIMEOUT = 20


def _is_public_hostname(hostname: str) -> bool:
    """Resolve hostname and reject private/link-local addresses."""
    try:
        infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror:
        return False
    for info in infos:
        addr = ipaddress.ip_address(info[4][0])
        if addr.is_private or addr.is_link_local or addr.is_loopback:
            return False
        if isinstance(addr, ipaddress.IPv6Address):
            if addr.is_multicast or addr.ipv4_mapped:
                mapped = addr.ipv4_mapped
                if mapped and (
                    mapped.is_private or mapped.is_link_local or mapped.is_loopback
                ):
                    return False
    return True


def _convert_urls(src: DocumentSource) -> str:  # noqa
    """Download remote document URLs from src.url, convert via MarkItDown.

    SSRF-safe: only http/https schemes, public hostnames, timeout + size limit.
    """
    urls = src.url or []
    doc_urls = []
    for u in urls:
        if not isinstance(u, str):
            continue
        parsed = urlparse(u)
        if parsed.scheme not in ("http", "https"):
            continue
        ext = os.path.splitext(parsed.path.lower())[1]
        if ext not in _DOCUMENT_EXTENSIONS:
            continue
        hostname = parsed.hostname
        if not hostname or not _is_public_hostname(hostname):
            continue
        doc_urls.append(u)

    if not doc_urls:
        return ""

    parts = []
    saved_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(max(_REQUEST_TIMEOUT, saved_timeout or 0))
    try:
        for url in doc_urls:
            tmp_path = None
            try:
                suffix = os.path.splitext(urlparse(url).path.lower())[1] or ".tmp"
                tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                tmp_path = tmp.name
                tmp.close()

                req = urllib.request.Request(
                    url, headers={"User-Agent": "JawafdehiAPI/1.0"}
                )
                with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
                    content_length = resp.headers.get("Content-Length")
                    if content_length and int(content_length) > _MAX_DOWNLOAD_BYTES:
                        logger.debug(
                            f"Skipping {url}: Content-Length {content_length} > {_MAX_DOWNLOAD_BYTES}"
                        )
                        continue
                    data = resp.read(_MAX_DOWNLOAD_BYTES + 1)
                    if len(data) > _MAX_DOWNLOAD_BYTES:
                        logger.debug(
                            f"Skipping {url}: download exceeds {_MAX_DOWNLOAD_BYTES} bytes"
                        )
                        continue
                    with open(tmp_path, "wb") as f:
                        f.write(data)

                doc_result = _MD_CONVERTER.convert(tmp_path)
                if doc_result and doc_result.text_content:
                    parts.append(doc_result.text_content)
            except Exception as e:
                logger.debug(f"URL download/conversion failed for {url}: {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
    finally:
        socket.setdefaulttimeout(saved_timeout)

    if parts:
        return " ".join(parts)[:1200]
    return ""


def _match_keywords(text: str, keyword_map: dict[str, list[str]]) -> list[str]:  # noqa
    """Match text against keyword maps. Returns matched tag names."""
    matched = []
    for tag, keywords in keyword_map.items():
        for kw in keywords:
            if kw.isascii() and all(c.isascii() for c in kw):
                if re.search(r"\b" + re.escape(kw) + r"\w*\b", text):
                    matched.append(tag)
                    break
            else:
                if kw in text:
                    matched.append(tag)
                    break
    return matched


def _detect_amount_tier(bigo) -> str | None:
    """Return human-readable Nepali amount string, or None if bigo is unknown.

    Examples:
        50000       → "~50 Hazar"
        477899      → "~4 Lakh 77 Hazar"
        49012323    → "~4 Crore 90 Lakh"
        1470000000  → "~147 Crore"
    """
    if bigo is None:
        return None

    arab = int(bigo // 1_000_000_000)
    remainder = bigo % 1_000_000_000
    crore = int(remainder // 10_000_000)
    remainder = remainder % 10_000_000
    lakh = int(remainder // 100_000)
    remainder = remainder % 100_000
    hazar = int(remainder // 1_000)

    parts = []
    if arab:
        parts.append(f"{arab} Arab")
    if crore:
        parts.append(f"{crore} Crore")
    if lakh and not arab:  # skip lakh if arab-level
        parts.append(f"{lakh} Lakh")
    if hazar and not arab and not crore:  # skip hazar if crore-level or above
        parts.append(f"{hazar} Hazar")

    if not parts:
        return "Under 1 Hazar"

    return "~" + " ".join(parts)


def _detect_court_context(case: Case) -> list[str]:
    tags = []
    tags.append("CIAA")
    tags.append("Corruption")
    if case.court_cases:
        for cc in case.court_cases:
            if isinstance(cc, str):
                if cc.startswith("special:"):
                    tags.append("Special Court")
                elif cc.startswith("supreme:"):
                    tags.append("Supreme Court")
    return tags


def _append_amount_tag(tags: list[str], bigo) -> list[str]:
    """Append bigo amount tag to tags list if available and not already present."""
    amount_tag = _detect_amount_tier(bigo)
    if amount_tag and amount_tag not in tags:
        tags.append(amount_tag)
    return tags


def _build_tag_selection_instructions() -> list[str]:
    """Build the common tag selection instructions for LLM prompts."""
    lines = []
    lines.append("Select the most appropriate tags from each category:")
    lines.append("")
    lines.append(f"Sector (choose 1-3): {', '.join(SECTOR_TAGS)}")
    lines.append(f"Corruption Type (choose 1-3): {', '.join(CORRUPTION_TYPE_TAGS)}")
    lines.append(f"Region (choose 1-2): {', '.join(REGION_TAGS)}")
    lines.append("")
    lines.append("Always include: CIAA, Corruption")
    lines.append("")
    lines.append("Return ONLY a JSON array of tag strings, nothing else.")
    lines.append(
        'Example: ["CIAA", "Corruption", "Local Government", "Bribery", "Kathmandu Valley"]'
    )
    return lines


def classify_case_rules(case: Case) -> list[str]:  # noqa
    """Rule-based tag classification for a CIAA case."""
    text = _collect_case_text(case)
    tags = []

    sectors = _match_keywords(text, SECTOR_KEYWORDS)
    tags.extend(sectors[:3])

    corruption_types = _match_keywords(text, CORRUPTION_TYPE_KEYWORDS)
    tags.extend(corruption_types[:3])

    regions = _match_keywords(text, REGION_KEYWORDS)
    tags.extend(regions[:2])

    amount_tier = _detect_amount_tier(case.bigo)
    if amount_tier is not None:
        tags.append(amount_tier)

    context_tags = _detect_court_context(case)
    tags.extend(context_tags)

    seen = set()
    unique = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    return unique


def build_llm_classification_prompt(case: Case) -> str:
    """Build a prompt for LLM-based tag classification from case metadata."""
    lines = []
    lines.append(
        "Classify the following Nepal corruption case with tags from the controlled vocabulary below."
    )
    lines.append("")
    lines.append(f"Case Title: {case.title}")
    if case.key_allegations:
        lines.append("Key Allegations:")
        for a in case.key_allegations:
            lines.append(f"  - {a}")
    if case.court_cases:
        lines.append(
            "Court Cases: "
            + ", ".join(c for c in case.court_cases if isinstance(c, str))
        )
    if case.bigo is not None:
        lines.append(f"Bigo (Disputed Amount): NPR {case.bigo:,}")
    lines.append("")
    lines.extend(_build_tag_selection_instructions())
    return "\n".join(lines)


def build_llm_classification_prompt_from_sources(case: Case, evidence_text: str) -> str:
    """Build a prompt for LLM-based tag classification using source documents."""
    lines = []
    lines.append(
        "Classify the following Nepal corruption case with tags from the controlled vocabulary below."
    )
    lines.append(
        "Use the source documents (press releases, court orders) as the primary evidence."
    )
    lines.append("")
    lines.append(f"Case Title: {case.title}")
    lines.append("")
    lines.append("Source Documents (press releases, court orders, evidence):")
    truncated = evidence_text[:1200]
    if len(evidence_text) > 1200:
        truncated += " [truncated]"
    lines.append(truncated)
    lines.append("")
    if case.court_cases:
        lines.append(
            "Court Cases: "
            + ", ".join(c for c in case.court_cases if isinstance(c, str))
        )
    if case.bigo is not None:
        lines.append(f"Bigo (Disputed Amount): NPR {case.bigo:,}")
    lines.append("")
    lines.extend(_build_tag_selection_instructions())
    return "\n".join(lines)


def parse_llm_response(response: str) -> list[str]:
    """Parse LLM response to extract a JSON list of tags."""
    response = response.strip()
    if response.startswith("```"):
        lines = response.split("\n")
        response = "\n".join(line for line in lines if not line.startswith("```"))
        response = response.strip()

    for match in re.finditer(r"\[[^\]]*\]", response):
        try:
            tags = json.loads(match.group())
            if isinstance(tags, list) and all(isinstance(t, str) for t in tags):
                return tags
        except json.JSONDecodeError:
            continue
    return []


def validate_tags(tags: list[str]) -> list[str]:
    """Filter tags to only include valid controlled vocabulary and deduplicate.

    Amount tier tags (~X Crore Y Lakh etc.) are dynamic and always pass through.
    """
    all_valid = set()
    for tag_list in [
        SECTOR_TAGS,
        CORRUPTION_TYPE_TAGS,
        REGION_TAGS,
        CONTEXT_TAGS,
    ]:
        all_valid.update(tag_list)

    valid = []
    seen = set()
    for t in tags:
        # Amount tags are dynamic (e.g. "~4 Crore 90 Lakh") — always valid
        is_amount = t.startswith("~") or t == "Under 1 Hazar"
        if (t in all_valid or is_amount) and t not in seen:
            seen.add(t)
            valid.append(t)
    return valid


class TagEnricher:
    """Service for enriching CIAA cases with tags via source documents + rule-based + LLM classification.

    Three-tier pipeline per case:
    1. Primary: LLM classification from source documents (evidence + DocumentSource)
    2. Fallback: Rule-based keyword matching on case metadata
    3. Default: Core tags only (CIAA, Corruption)

    Includes circuit breaker protection, retry with exponential backoff,
    and Prometheus-compatible metrics recording.
    """

    def __init__(self, use_llm: bool = True, llm_client=None, model: str = "unknown"):
        self.use_llm = use_llm
        self._llm_client = llm_client
        self._llm_service = None
        self._model = model

        from cases.circuit_breaker import CircuitBreaker

        self._source_llm_cb = CircuitBreaker(name="source_llm")
        self._metadata_llm_cb = CircuitBreaker(name="metadata_llm")

    def _invoke_llm(self, prompt: str, circuit_name: str = "unknown") -> str:
        from cases.observability import record_llm_outcome
        from cases.retry import retry_with_backoff

        def _call() -> str:
            if self._llm_client is not None:
                logger.debug("  Using CLI-provided LLM client (bypassing DB LLMProvider)")
                response = self._llm_client.invoke(prompt)
                if hasattr(response, "content"):
                    return response.content
                return str(response)

            if self._llm_service is None:
                from caseworker.services import LLMService

                self._llm_service = LLMService()

            logger.debug("  Using DB LLMProvider")
            llm = self._llm_service.get_llm()
            return self._llm_service._call_llm(llm, prompt)

        try:
            result = retry_with_backoff(_call, max_retries=3, base_seconds=1.0, max_seconds=30.0)
            record_llm_outcome(True, model=self._model, command=circuit_name)
            return result
        except Exception:
            record_llm_outcome(False, model=self._model, command=circuit_name)
            raise

    def enrich_case(
        self, case: Case, force: bool = False, case_num: int = 0, total_cases: int = 0
    ) -> dict:  # noqa
        """Enrich a single case with tags. Returns dict with status, tags, and tier."""
        from cases.observability import track_pipeline_duration

        case_number = None
        if case.court_cases:
            for entry in case.court_cases:
                if isinstance(entry, str) and ":" in entry:
                    case_number = entry.split(":", 1)[1]
                    break

        if case_num and total_cases:
            case_num_str = f" (#{case_number})" if case_number else ""
            logger.info(
                f"[{case_num}/{total_cases}] Processing {case.case_id}{case_num_str} — {case.title[:70] if case.title else 'No title'}"
            )
        else:
            logger.info(f"Processing {case.case_id}...")

        logger.info(f"  Title: {case.title[:70] if case.title else 'No title'}")
        if case_number:
            logger.info(f"  Case Number: {case_number}")
        logger.info(f"  Bigo:  bigo={case.bigo}")

        if case.tags and len(case.tags) > 0 and not force:
            logger.info("  Already tagged, skipping")
            return {
                "status": "skipped",
                "tags": case.tags,
                "tier": "already_tagged",
                "reason": "already has tags",
            }

        evidence_text = ""
        has_evidence = False
        if self.use_llm:
            evidence_text = _collect_evidence_text(case)
            has_evidence = bool(evidence_text.strip())

        tier = "rule_based"

        if self.use_llm and has_evidence:
            with track_pipeline_duration(tier="source_llm", command="enrich_ciaa_tags"):
                logger.info("  Attempting source_llm classification...")
                try:
                    llm_tags = self._classify_with_llm_from_sources(case, evidence_text)
                    if llm_tags:
                        validated = validate_tags(list(dict.fromkeys(llm_tags)))
                        if validated:
                            if len(validated) >= 3:
                                all_tags = validated
                                logger.info(
                                    f"  + source_llm succeeded: {len(all_tags)} tags"
                                )
                            else:
                                rule_tags = classify_case_rules(case)
                                all_tags = validate_tags(
                                    list(dict.fromkeys(validated + rule_tags))
                                )
                                logger.info(
                                    f"  + source_llm augmented: {len(all_tags)} tags"
                                )
                            logger.info(
                                f"+ Enriched {case.case_id} (source_llm): {all_tags}"
                            )
                            all_tags = _append_amount_tag(all_tags, case.bigo)
                            return {
                                "status": "enriched",
                                "tags": all_tags,
                                "tier": "source_llm",
                                "reason": "",
                            }
                        else:
                            logger.warning(
                                "  - source_llm returned no valid tags, falling back"
                            )
                    else:
                        logger.warning("  - source_llm returned no tags, falling back")
                except Exception as e:
                    logger.warning(f"  - source_llm failed: {str(e)[:120]}")

        tags = classify_case_rules(case)

        llm_invoked = False
        llm_contributed = False
        if self.use_llm and len(tags) < 5:
            with track_pipeline_duration(tier="metadata_llm", command="enrich_ciaa_tags"):
                logger.info("  Attempting metadata_llm classification...")
                llm_invoked = True
                try:
                    llm_tags = self._classify_with_llm(case)
                    if llm_tags:
                        new_tags = list(dict.fromkeys(tags + llm_tags))
                        llm_contributed = len(new_tags) > len(tags)
                        all_tags = new_tags
                        logger.info(f"  + metadata_llm succeeded: {len(all_tags)} tags")
                    else:
                        all_tags = tags
                        logger.info("  - metadata_llm returned no tags, using rule-based")
                except Exception as e:
                    logger.warning(f"  - metadata_llm failed: {str(e)[:120]}")
                    all_tags = tags
        else:
            all_tags = tags

        all_tags = validate_tags(all_tags)

        if not all_tags:
            all_tags = ["CIAA", "Corruption"]

        if llm_invoked and llm_contributed:
            tier = "metadata_llm"
        else:
            tier = "rule_based"
        all_tags = _append_amount_tag(all_tags, case.bigo)
        logger.info(f"+ Enriched {case.case_id} ({tier}): {all_tags}")
        return {"status": "enriched", "tags": all_tags, "tier": tier, "reason": ""}

    def _classify_with_llm(self, case: Case) -> list[str]:
        """Use LLM to classify a case from metadata. Returns list of tag strings."""
        prompt = build_llm_classification_prompt(case)
        response = self._metadata_llm_cb.call(
            lambda: self._invoke_llm(prompt, circuit_name="metadata_llm")
        )
        return parse_llm_response(response)

    def _classify_with_llm_from_sources(
        self, case: Case, evidence_text: str
    ) -> list[str]:
        """Use LLM to classify a case from source documents. Returns list of tag strings."""
        prompt = build_llm_classification_prompt_from_sources(case, evidence_text)
        response = self._source_llm_cb.call(
            lambda: self._invoke_llm(prompt, circuit_name="source_llm")
        )
        return parse_llm_response(response)

    def enrich_cases(self, cases, force: bool = False, dry_run: bool = False) -> dict:
        """Enrich multiple cases. Returns stats dict."""
        stats = {
            "total": 0,
            "enriched": 0,
            "skipped": 0,
            "failed": 0,
            "source_llm": 0,
            "metadata_llm": 0,
            "rule_based": 0,
        }

        cases_list = list(cases)
        total_cases = len(cases_list)

        for idx, case in enumerate(cases_list, start=1):
            stats["total"] += 1
            try:
                result = self.enrich_case(
                    case, force=force, case_num=idx, total_cases=total_cases
                )
                if result["status"] == "skipped":
                    stats["skipped"] += 1
                    logger.debug(f"Skipped {case.case_id}: {result['reason']}")
                else:
                    stats["enriched"] += 1
                    tier = result.get("tier", "unknown")
                    if tier in stats:
                        stats[tier] += 1
                    if not dry_run:
                        case.tags = result["tags"]
                        case.save(update_fields=["tags", "updated_at"])
                        logger.info(
                            f"✓ Saved {case.case_id} with {len(result['tags'])} tags"
                        )
                    else:
                        logger.info(f"  [DRY RUN] Would save: {result['tags']}")
            except Exception as e:
                stats["failed"] += 1
                logger.exception(f"Failed to enrich {case.case_id}: {e}")
        return stats
