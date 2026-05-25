from django.urls import path

from cases.health import enrichment_health, enrichment_metrics

urlpatterns = [
    path("", enrichment_health, name="enrichment-health"),
    path("metrics", enrichment_metrics, name="enrichment-metrics"),
]
