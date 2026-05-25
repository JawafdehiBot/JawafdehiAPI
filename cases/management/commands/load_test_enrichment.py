"""Load test: 50 concurrent case enrichment operations.

Usage:
    python manage.py load_test_enrichment --concurrency 50 [--case-count 50]

Requires a running Django app with database access. This command exercises
the TagEnricher pipeline under concurrent load and prints timing metrics.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from django.core.management.base import BaseCommand

from cases.models import Case, CaseState
from cases.services.tag_enricher import TagEnricher
from cases.observability import export_textfile, pipeline_duration


class Command(BaseCommand):
    help = "Load test the enrichment pipeline with concurrent case processing."

    def add_arguments(self, parser):
        parser.add_argument(
            "--concurrency",
            type=int,
            default=50,
            help="Number of concurrent workers (default 50).",
        )
        parser.add_argument(
            "--case-count",
            type=int,
            default=50,
            help="Number of cases to process (default 50).",
        )
        parser.add_argument(
            "--no-llm",
            action="store_true",
            help="Disable LLM calls (rule-based only).",
        )
        parser.add_argument(
            "--metrics-file",
            type=str,
            default=None,
            help="Path to write Prometheus textfile after the run.",
        )

    def handle(self, *args, **options):
        concurrency = options["concurrency"]
        case_count = options["case_count"]
        use_llm = not options["no_llm"]
        metrics_file = options.get("metrics_file")

        self.stdout.write(
            self.style.WARNING(
                f"Load test: {concurrency} concurrent workers, "
                f"{case_count} cases, LLM={'on' if use_llm else 'off'}"
            )
        )

        cases = list(
            Case.objects.filter(state=CaseState.DRAFT).order_by("?")[:case_count]
        )
        if not cases:
            self.stdout.write(self.style.ERROR("No DRAFT cases found."))
            return

        self.stdout.write(f"Loaded {len(cases)} DRAFT cases for testing.")

        enricher = TagEnricher(use_llm=use_llm)

        start = time.monotonic()
        completed = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {}
            for idx, case in enumerate(cases):
                futures[
                    executor.submit(
                        self._enrich_one, enricher, case, idx + 1, len(cases)
                    )
                ] = case
            for future in as_completed(futures):
                case = futures[future]
                try:
                    result = future.result()
                    if result.get("status") == "failed":
                        failed += 1
                    else:
                        completed += 1
                except Exception:
                    failed += 1

        elapsed = time.monotonic() - start
        snap = pipeline_duration.snapshot()

        self.stdout.write("")
        self.stdout.write("=" * 60)
        self.stdout.write(self.style.SUCCESS("LOAD TEST RESULTS"))
        self.stdout.write("=" * 60)
        self.stdout.write(f"Concurrency:       {concurrency}")
        self.stdout.write(f"Cases attempted:   {len(cases)}")
        self.stdout.write(f"Completed:         {completed}")
        self.stdout.write(f"Failed:            {failed}")
        self.stdout.write(f"Total wall time:   {elapsed:.2f}s")
        self.stdout.write(f"Throughput:        {len(cases) / elapsed:.2f} cases/s")
        if snap["count"] > 0:
            self.stdout.write(f"Avg pipeline time: {snap['sum'] / snap['count']:.2f}s")
            self.stdout.write(f"Pipeline count:    {snap['count']}")
        self.stdout.write("=" * 60)

        if metrics_file:
            export_textfile(metrics_file)
            self.stdout.write(f"Metrics written to {metrics_file}")

    def _enrich_one(self, enricher, case, case_num, total):
        try:
            return enricher.enrich_case(
                case, force=False, case_num=case_num, total_cases=total
            )
        except Exception as e:
            return {"status": "failed", "reason": str(e)}
