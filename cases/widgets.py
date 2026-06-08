import json
from json import JSONDecodeError

from django.core.exceptions import ValidationError
from django.core.validators import URLValidator
from django.forms.fields import Field
from django.forms.widgets import Textarea, Widget
from django.template.loader import render_to_string
from django.utils.safestring import mark_safe
from nes.core.identifiers.validators import validate_entity_id

from .models import SourceLinkRole


class EasyMDEWidget(Textarea):
    class Media:
        css = {
            "all": ("https://cdn.jsdelivr.net/npm/easymde@2.19.0/dist/easymde.min.css",)
        }
        js = ("https://cdn.jsdelivr.net/npm/easymde@2.19.0/dist/easymde.min.js",)

    def __init__(self, attrs=None):
        default_attrs = {
            "cols": 80,
            "rows": 25,
            "data-easymde": "true",
        }
        if attrs:
            default_attrs.update(attrs)
        super().__init__(default_attrs)

    def render(self, name, value, attrs=None, renderer=None):
        textarea_html = super().render(name, value, attrs, renderer)
        final_attrs = self.build_attrs(self.attrs, attrs)
        widget_id = final_attrs.get("id", f"id_{name}")
        init_script = f"""
<script>
(function() {{
    var el = document.getElementById({json.dumps(widget_id)});
    if (el && !el._easymde_initialized) {{
        el._easymde_initialized = true;
        new EasyMDE({{
            element: el,
            spellChecker: false,
            status: ["lines", "words", "cursor"],
            toolbar: [
                "bold", "italic", "strikethrough", "heading", "|",
                "quote", "unordered-list", "ordered-list", "|",
                "link", "image", "table", "horizontal-rule", "|",
                "preview", "side-by-side", "fullscreen", "|",
                "guide"
            ],
            renderingConfig: {{
                singleLineBreaks: false,
                codeSyntaxHighlighting: true,
            }},
            placeholder: "Write in Markdown...",
            autosave: {{
                enabled: false,
            }},
            minHeight: "400px",
            maxHeight: "600px",
        }});
    }}
}})();
</script>
"""
        return mark_safe(textarea_html + init_script)


class ToastUIEditorWidget(Textarea):
    """
    Rich text Markdown editor powered by Toast UI Editor (TOAST UI Editor 3.x).
    Renders a WYSIWYG editing surface that auto-syncs Markdown to a hidden
    <textarea> on form submit. Drop-in replacement for EasyMDEWidget with
    native Word / Office copy-paste handling.
    """

    class Media:
        css = {
            "all": (
                "https://uicdn.toast.com/editor/3.2.2/toastui-editor.min.css",
                "https://uicdn.toast.com/editor-plugin-table-merged-cell/3.0.2/toastui-editor-plugin-table-merged-cell.min.css",
            )
        }
        js = (
            "https://uicdn.toast.com/editor/3.2.2/toastui-editor-all.min.js",
            "https://uicdn.toast.com/editor-plugin-table-merged-cell/3.0.2/toastui-editor-plugin-table-merged-cell.min.js",
        )

    def __init__(self, attrs=None):
        default_attrs = {
            "rows": 1,
            "data-toastui-editor": "true",
        }
        if attrs:
            default_attrs.update(attrs)
        super().__init__(default_attrs)

    def render(self, name, value, attrs=None, renderer=None):
        textarea_html = super().render(name, value, attrs, renderer)
        final_attrs = self.build_attrs(self.attrs, attrs)
        widget_id = final_attrs.get("id", f"id_{name}")
        editor_container_id = f"{widget_id}_editor"
        initial_md = json.dumps(value if value else "")

        init_script = f"""
<div id="{editor_container_id}"></div>
<script>
(function() {{
    var containerEl = document.getElementById({json.dumps(editor_container_id)});
    var textareaEl = document.getElementById({json.dumps(widget_id)});
    if (containerEl && textareaEl && !containerEl._tui_initialized) {{
        containerEl._tui_initialized = true;

        var editor = new toastui.Editor({{
            el: containerEl,
            height: '500px',
            minHeight: '400px',
            initialEditType: 'wysiwyg',
            previewStyle: 'vertical',
            initialValue: {initial_md},
            usageStatistics: false,
            toolbarItems: [
                ['heading', 'bold', 'italic', 'strike'],
                ['hr', 'quote'],
                ['ul', 'ol', 'task', 'indent', 'outdent'],
                ['table', 'image', 'link'],
                ['code', 'codeblock'],
                ['scrollSync'],
            ],
            plugins: [toastui.Editor.plugin.tableMergedCell],
            placeholder: 'Write in Markdown\u2026',
            autofocus: false,
            previewHighlight: true,
        }});

        textareaEl.style.display = 'none';

        var form = textareaEl.closest('form');
        if (form) {{
            form.addEventListener('submit', function() {{
                textareaEl.value = editor.getMarkdown();
            }});
        }}
    }}
}})();
</script>
"""
        return mark_safe(textarea_html + init_script)


class BaseMultiWidget(Widget):
    template_name = None

    class Media:
        css = {"all": ("cases/css/widgets.css",)}
        js = ("cases/js/widgets.js",)

    def get_context(self, name, value, attrs):
        if value is None:
            value = []
        elif isinstance(value, str):
            value = json.loads(value) if value else []

        final_attrs = self.build_attrs(self.attrs, attrs)
        widget_id = final_attrs.get("id", name)

        return {
            "widget_id": widget_id,
            "name": name,
            "values": value,
            "values_json": json.dumps(value),
        }

    def render(self, name, value, attrs=None, renderer=None):
        context = self.get_context(name, value, attrs)
        return mark_safe(render_to_string(self.template_name, context))

    def value_from_datadict(self, data, files, name):
        value = data.get(name, "[]")
        if isinstance(value, list):
            return value
        try:
            return json.loads(value) if value else []
        except (json.JSONDecodeError, TypeError, ValueError):
            return []


class MultiEntityIDWidget(BaseMultiWidget):
    template_name = "cases/widgets/multi_entity_widget.html"


class MultiEntityIDField(Field):
    widget = MultiEntityIDWidget

    def to_python(self, value):
        if value in self.empty_values:
            return []
        if isinstance(value, list):
            return value
        try:
            return json.loads(value) if value else []
        except (json.JSONDecodeError, TypeError, ValueError):
            return []

    def validate(self, value):
        super().validate(value)
        for entity_id in value:
            try:
                validate_entity_id(entity_id)
            except ValueError as e:
                raise ValidationError(str(e))


class MultiTextWidget(BaseMultiWidget):
    template_name = "cases/widgets/multi_text_widget.html"

    def __init__(self, attrs=None, button_label=None):
        super().__init__(attrs)
        self.button_label = button_label or "Add Item"

    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        context["button_label"] = self.button_label
        return context


class MultiTextField(Field):
    def __init__(self, *args, button_label="Add Item", **kwargs):
        self.button_label = button_label
        super().__init__(*args, **kwargs)
        self.widget = MultiTextWidget(button_label=button_label)

    def to_python(self, value):
        if value in self.empty_values:
            return []
        if isinstance(value, list):
            return value
        try:
            return json.loads(value) if value else []
        except (json.JSONDecodeError, TypeError, ValueError):
            return []

    def validate(self, value):
        super().validate(value)
        # Only validate non-empty if field is required
        if self.required and (not value or len(value) < 1):
            raise ValidationError("This field is required.")


class MultiTimelineWidget(BaseMultiWidget):
    template_name = "cases/widgets/multi_timeline_widget.html"


class MultiTimelineField(Field):
    widget = MultiTimelineWidget

    def to_python(self, value):
        if value in self.empty_values:
            return []
        if isinstance(value, list):
            return value
        try:
            return json.loads(value) if value else []
        except (json.JSONDecodeError, TypeError, ValueError):
            return []


class MultiEvidenceWidget(BaseMultiWidget):
    template_name = "cases/widgets/multi_evidence_widget.html"

    def __init__(self, attrs=None, sources=None):
        super().__init__(attrs)
        self.sources = sources or []

    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        context["sources"] = self.sources
        return context


class MultiEvidenceField(Field):
    def __init__(self, *args, **kwargs):
        self.sources = kwargs.pop("sources", [])
        super().__init__(*args, **kwargs)
        self.widget = MultiEvidenceWidget(sources=self.sources)

    def to_python(self, value):
        if value in self.empty_values:
            return []
        if isinstance(value, list):
            return value
        try:
            return json.loads(value) if value else []
        except (json.JSONDecodeError, TypeError, ValueError):
            return []


class MultiURLWidget(BaseMultiWidget):
    template_name = "cases/widgets/multi_url_widget.html"

    def __init__(self, attrs=None, button_label=None):
        super().__init__(attrs)
        self.button_label = button_label or "Add URL"

    def get_context(self, name, value, attrs):
        """
        Override to handle invalid JSON gracefully and normalise str entries
        to ``{link, role}`` dict form.
        """
        if value is None:
            value = []
        elif isinstance(value, str):
            try:
                value = json.loads(value) if value else []
            except (JSONDecodeError, TypeError):
                # Invalid JSON - use empty list so form can render and show validation error
                value = []

        # Normalise plain strings to dict format
        normalised = []
        for item in value:
            if isinstance(item, str):
                normalised.append({"link": item, "role": None})
            elif isinstance(item, dict):
                normalised.append(item)
            else:
                normalised.append(item)
        value = normalised

        final_attrs = self.build_attrs(self.attrs, attrs)
        widget_id = final_attrs.get("id", name)

        context = {
            "widget_id": widget_id,
            "name": name,
            "values": value,
            "values_json": json.dumps(value),
            "button_label": self.button_label,
        }
        return context

    def value_from_datadict(self, data, files, name):
        """
        Extract value from form data without silently converting parse errors to [].
        Let MultiURLField.to_python() handle JSON parsing and raise ValidationError.
        """
        value = data.get(name, "[]")
        return value if value is not None else "[]"


class MultiURLField(Field):
    def __init__(self, *args, button_label="Add URL", **kwargs):
        self.button_label = button_label
        super().__init__(*args, **kwargs)
        self.widget = MultiURLWidget(button_label=button_label)

    def to_python(self, value):
        if value in self.empty_values:
            return []
        if isinstance(value, list):
            return value
        try:
            parsed = json.loads(value) if value else []
        except (JSONDecodeError, TypeError) as err:
            raise ValidationError("Invalid URL payload format.") from err

        if not isinstance(parsed, list):
            raise ValidationError("Expected a list of URLs.")
        return parsed

    def validate(self, value):
        super().validate(value)
        if value:
            validator = URLValidator()
            for item in value:
                if isinstance(item, dict):
                    link = item.get("link", "")
                    role = item.get("role")
                    if not isinstance(link, str) or not link.strip():
                        raise ValidationError(
                            "Each URL dict must contain a non-blank 'link' string."
                        )
                    if role is not None:
                        try:
                            SourceLinkRole(role)
                        except ValueError:
                            raise ValidationError(
                                f"Invalid role '{role}'. Must be one of: "
                                f"{', '.join(e.value for e in SourceLinkRole)}"
                            )
                    normalized = link.strip()
                elif isinstance(item, str):
                    normalized = item.strip()
                else:
                    raise ValidationError(
                        f"Invalid URL type: expected string or dict, got "
                        f"{type(item).__name__} ({item!r})"
                    )

                if not normalized:
                    raise ValidationError(
                        f"URL cannot be blank or whitespace-only: {item!r}"
                    )

                try:
                    validator(normalized)
                except ValidationError as err:
                    raise ValidationError(f"Invalid URL: {item}") from err


class MultiCourtCaseWidget(BaseMultiWidget):
    template_name = "cases/widgets/multi_court_case_widget.html"

    def __init__(self, attrs=None, button_label=None, court_choices=None):
        super().__init__(attrs)
        self.button_label = button_label or "Add Court Case"
        self.court_choices = court_choices or []

    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        context["button_label"] = self.button_label
        context["court_choices"] = self.court_choices

        # JSON-encode court_choices for JavaScript
        context["court_choices_json"] = json.dumps(self.court_choices)

        # Parse court case values into structured format for template
        # Use context["values"] (normalized by BaseMultiWidget) instead of raw value parameter
        parsed_values = []
        if context["values"]:
            for item in context["values"]:
                if isinstance(item, str) and ":" in item:
                    court_id, case_number = item.split(":", 1)
                    parsed_values.append(
                        {
                            "court_id": court_id,
                            "case_number": case_number,
                            "full_value": item,
                        }
                    )

        context["parsed_values"] = parsed_values
        return context


class MultiCourtCaseField(Field):
    def __init__(
        self, *args, button_label="Add Court Case", court_choices=None, **kwargs
    ):
        self.button_label = button_label
        self.court_choices = court_choices or []
        super().__init__(*args, **kwargs)
        self.widget = MultiCourtCaseWidget(
            button_label=button_label, court_choices=court_choices
        )

    def to_python(self, value):
        if value in self.empty_values:
            return []
        if isinstance(value, list):
            return value
        try:
            return json.loads(value) if value else []
        except (JSONDecodeError, TypeError, ValueError):
            return []

    def validate(self, value):
        super().validate(value)
        # Import here to avoid circular dependency
        from .validators import validate_court_cases

        try:
            validate_court_cases(value)
        except ValidationError:
            raise
