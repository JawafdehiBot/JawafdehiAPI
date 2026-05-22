"""
Django management command to enrich CIAA corruption cases with related news
articles and image links.

Phase 2d of CIAA FY 080/081 Case Enrichment pipeline.
Searches for news articles, verifies relevance with LLM, and stores as
DocumentSource records with MEDIA_NEWS type.

Usage::

    python manage.py enrich_ciaa_news_articles --dry-run
    python manage.py enrich_ciaa_news_articles --priority --limit 5
    python manage.py enrich_ciaa_news_articles --case-id case-0123
"""

import logging
import os

from django.core.management.base import BaseCommand

from cases.models import Case, CaseType, CaseState
from cases.services.news_enricher import NewsEnricher, enrich_cases_batch
from cases.services.priority_case_loader import filter_by_priority, load_priority_cases

logger = logging.getLogger(__name__)

_LOGGERS_TO_OVERRIDE = (
    "cases.services.news_enricher",
    "cases.management.commands.enrich_ciaa_news_articles",
)


class Command(BaseCommand):
    help = "Enrich CIAA cases with related news articles and image links"

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Preview without saving to database",
        )
        parser.add_argument(
            "--case-id",
            type=str,
            help="Process a specific case by case_id",
        )
        parser.add_argument(
            "--priority",
            action="store_true",
            help="Enrich only cases in the priority case list",
        )
        parser.add_argument(
            "--all",
            action="store_true",
            dest="all_cases",
            help="Enrich all DRAFT CIAA cases (explicit, same as default)",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Re-enrich cases that already have news evidence links",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Limit number of cases to process",
        )
        parser.add_argument(
            "--max-articles",
            type=int,
            default=5,
            help="Maximum articles per case (default: 5)",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose debug logging",
        )
        parser.add_argument(
            "--llm-model",
            type=str,
            default=None,
            help="LLM model identifier (default: from env or gpt-4.5)",
        )
        parser.add_argument(
            "--llm-base-url",
            type=str,
            default=None,
            help="LLM API base URL (default: JAWAFDEHI_LLM_PROXY_URL or llm-proxy)",
        )
        parser.add_argument(
            "--llm-api-key",
            type=str,
            default=None,
            help="LLM API key (defaults to JAWAFDEHI_LLM_API_KEY or ANTHROPIC_API_KEY)",
        )
        parser.add_argument(
            "--search-delay",
            type=float,
            default=1.0,
            help="Delay in seconds between search queries (default: 1.0)",
        )
        parser.add_argument(
            "--fetch-delay",
            type=float,
            default=0.5,
            help="Delay in seconds between article fetches (default: 0.5)",
        )

    def _validate_numeric_options(self, options):
        """Validate numeric CLI options are non-negative. Returns error message or None."""
        for name in ("limit", "max_articles"):
            val = options.get(name)
            if val is not None and val < 0:
                return f"--{name} must be a non-negative integer"
        search_delay = options.get("search_delay", 1.0)
        fetch_delay = options.get("fetch_delay", 0.5)
        if search_delay < 0 or fetch_delay < 0:
            return "--search-delay and --fetch-delay must be non-negative"
        return None

    def _resolve_llm_config(self, options):
        """Resolve LLM model, base URL, and API key from options and environment."""
        model = options.get("llm_model") or os.environ.get(
            "JAWAFDEHI_LLM_MODEL", "gpt-4.5"
        )
        base_url = options.get("llm_base_url") or os.environ.get(
            "JAWAFDEHI_LLM_PROXY_URL", "https://llm-proxy.jawafdehi.org/v1"
        )
        api_key = (
            options.get("llm_api_key")
            or os.environ.get("JAWAFDEHI_LLM_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        return model, base_url, api_key

    def _build_case_queryset(self, case_id, priority, all_cases_flag):
        """Build the case queryset based on CLI options."""
        if case_id:
            cases = Case.objects.filter(
                case_id=case_id,
                case_type=CaseType.CORRUPTION,
                state__in=[CaseState.DRAFT, CaseState.IN_REVIEW],
            )
        else:
            cases = Case.objects.filter(
                case_type=CaseType.CORRUPTION,
                state__in=[CaseState.DRAFT, CaseState.IN_REVIEW],
            ).order_by("created_at")

            if priority:
                priority_list = load_priority_cases()
                logger.info(
                    "Priority mode: loaded %d case numbers across all fiscal years",
                    len(priority_list),
                )
                cases = filter_by_priority(cases, priority_list)
            elif not all_cases_flag:
                logger.info(
                    "Processing all DRAFT and IN_REVIEW CIAA cases (default). "
                    "Use --all to make this explicit or --priority to filter."
                )
        return cases

    def _setup_plain_text_logging(self):
        """Override Django's JSON logging with plain-text output for this command.

        Matches the pattern used by enrich_ciaa_allegations. For each logger
        used by this enrichment pipeline, installs a plain-text StreamHandler
        on self.stdout and disables propagation to the root logger, preventing
        JSON-structured logs from leaking through.
        """
        fmt = logging.Formatter("%(levelname)s: %(message)s")
        for name in _LOGGERS_TO_OVERRIDE:
            child_logger = logging.getLogger(name)
            child_logger.handlers.clear()
            handler = logging.StreamHandler(self.stdout)
            handler.setFormatter(fmt)
            child_logger.addHandler(handler)
            child_logger.propagate = False
            child_logger.setLevel(
                logging.DEBUG if logger.isEnabledFor(logging.DEBUG) else logging.INFO
            )

    def handle(self, *args, **options):
        dry_run = options["dry_run"]
        case_id = options.get("case_id")
        priority = options["priority"]
        all_cases_flag = options.get("all_cases")
        force = options["force"]
        limit = options.get("limit")
        max_articles = options.get("max_articles", 5)
        verbose = options.get("verbose")

        error = self._validate_numeric_options(options)
        if error:
            self.stderr.write(self.style.ERROR(error))
            return

        if verbose:
            logger.setLevel(logging.DEBUG)

        self._setup_plain_text_logging()

        if priority and case_id:
            self.stderr.write(
                self.style.ERROR("--priority and --case-id are mutually exclusive")
            )
            return

        if dry_run:
            logger.warning("DRY-RUN MODE: No changes will be saved")

        model, base_url, api_key = self._resolve_llm_config(options)

        if not dry_run and not api_key:
            self.stderr.write(
                self.style.ERROR(
                    "No LLM API key found. Set JAWAFDEHI_LLM_API_KEY, "
                    "ANTHROPIC_API_KEY, or use --llm-api-key."
                )
            )
            return

        if dry_run and not api_key:
            logger.warning(
                "No LLM API key configured — article relevance verification disabled. "
                "Set JAWAFDEHI_LLM_API_KEY, ANTHROPIC_API_KEY, or use --llm-api-key."
            )

        cases = self._build_case_queryset(case_id, priority, all_cases_flag)

        if limit is not None:
            cases = cases[:limit]

        total = cases.count()
        if total == 0:
            self.stdout.write("No cases to process")
            self._print_summary({}, dry_run)
            return

        logger.info(
            "Processing %d cases (max %d articles per case)", total, max_articles
        )
        logger.info("LLM: %s @ %s", model, base_url)

        enricher = NewsEnricher(
            llm_model=model,
            llm_base_url=base_url,
            llm_api_key=api_key,
            max_articles_per_case=max_articles,
            search_delay=options.get("search_delay", 1.0),
            fetch_delay=options.get("fetch_delay", 0.5),
            verbose=verbose,
        )

        stats = enrich_cases_batch(
            enricher=enricher,
            cases=cases.iterator(),
            dry_run=dry_run,
            force=force,
        )

        self._print_summary(stats, dry_run)

    def _print_summary(self, stats: dict, dry_run: bool):
        self.stdout.write("")
        self.stdout.write("=" * 60)
        self.stdout.write("NEWS ARTICLE ENRICHMENT SUMMARY")
        self.stdout.write("=" * 60)
        if dry_run:
            self.stdout.write(self.style.WARNING("DRY-RUN MODE (no changes saved)"))

        total = stats.get("total", 0)
        self.stdout.write(f"Total cases:     {total}")
        if total > 0:
            self.stdout.write(f"Processed:       {stats.get('processed', 0)}")
            self.stdout.write(f"Skipped:         {stats.get('skipped', 0)}")
            self.stdout.write("")
            self.stdout.write(f"Searches:        {stats.get('searched', 0)}")
            self.stdout.write(f"Articles fetched:{stats.get('fetched', 0)}")
            self.stdout.write(f"Accepted:        {stats.get('accepted', 0)}")
            self.stdout.write(f"Rejected:        {stats.get('rejected', 0)}")
            self.stdout.write(f"Already linked:  {stats.get('already_linked', 0)}")
            self.stdout.write(f"New sources:     {stats.get('new_sources', 0)}")
            self.stdout.write(f"Errors:          {stats.get('errors', 0)}")
        self.stdout.write("=" * 60)
