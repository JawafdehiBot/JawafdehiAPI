"""Convert existing string URL entries to dict format within the JSON url list."""

from django.db import migrations


def migrate_url_strs_to_dicts(apps, schema_editor):
    """
    Convert each str entry in the url JSON list to ``{link: str, role: null}``.

    Existing dict entries are left intact (already in the correct format).
    """
    DocumentSource = apps.get_model("cases", "DocumentSource")
    db_alias = schema_editor.connection.alias
    to_update = []

    for source in DocumentSource.objects.using(db_alias).only("id", "url").iterator():
        url_list = source.url
        if not isinstance(url_list, list):
            continue

        changed = False
        converted = []
        for item in url_list:
            if isinstance(item, str):
                converted.append({"link": item, "role": None})
                changed = True
            elif isinstance(item, dict):
                # Ensure role key exists for consistency
                if "role" not in item:
                    item["role"] = None
                    changed = True
                converted.append(item)
            else:
                converted.append(item)

        if changed:
            source.url = converted
            to_update.append(source)

    if to_update:
        DocumentSource.objects.using(db_alias).bulk_update(
            to_update, ["url"], batch_size=500
        )


def reverse_url_dicts_to_strs(apps, schema_editor):
    """
    Convert dict entries back to plain strings (for rollback).

    Each ``{link, role}`` dict becomes just the ``link`` string value.
    """
    DocumentSource = apps.get_model("cases", "DocumentSource")
    db_alias = schema_editor.connection.alias
    to_update = []

    for source in DocumentSource.objects.using(db_alias).only("id", "url").iterator():
        url_list = source.url
        if not isinstance(url_list, list):
            continue

        changed = False
        converted = []
        for item in url_list:
            if isinstance(item, dict):
                converted.append(item.get("link", ""))
                changed = True
            else:
                converted.append(item)

        if changed:
            source.url = converted
            to_update.append(source)

    if to_update:
        DocumentSource.objects.using(db_alias).bulk_update(
            to_update, ["url"], batch_size=500
        )


class Migration(migrations.Migration):

    dependencies = [
        ("cases", "0024_alter_chat_user_identity"),
    ]

    operations = [
        migrations.RunPython(
            migrate_url_strs_to_dicts, reverse_url_dicts_to_strs, atomic=True
        ),
    ]
