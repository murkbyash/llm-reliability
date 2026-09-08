"""Helpers and attribute decoders for OpenTelemetry and semantic convention parsers."""

from typing import Any


class OTelAttributeParser:
    """Decodes standard OpenTelemetry attribute lists and dictionary representations."""

    @staticmethod
    def parse_attributes(raw_attrs: Any) -> dict[str, Any]:
        """Convert standard OTel key-value attribute structures into a standard Python dictionary.

        Supports both standard OTel protobuf/JSON formats:
          [{"key": "k", "value": {"stringValue": "v"}}, ...]
        and flattened key-value dictionaries:
          {"k": "v", ...}
        """
        if not raw_attrs:
            return {}

        if isinstance(raw_attrs, dict):
            result: dict[str, Any] = {}
            for k, v in raw_attrs.items():
                result[k] = OTelAttributeParser._unwrap_otel_value(v)
            return result

        if isinstance(raw_attrs, list):
            result = {}
            for item in raw_attrs:
                if isinstance(item, dict) and "key" in item:
                    key = item["key"]
                    val_container = item.get("value", item.get("val", {}))
                    result[key] = OTelAttributeParser._unwrap_otel_value(val_container)
            return result

        return {}

    @staticmethod
    def _unwrap_otel_value(val_wrapper: Any) -> Any:
        """Unwrap an OTel typed value object like {"stringValue": "foo"}."""
        if not isinstance(val_wrapper, dict):
            return val_wrapper

        for field in (
            "stringValue",
            "string_value",
            "intValue",
            "int_value",
            "doubleValue",
            "double_value",
            "boolValue",
            "bool_value",
        ):
            if field in val_wrapper:
                return val_wrapper[field]

        if "arrayValue" in val_wrapper or "array_value" in val_wrapper:
            arr = val_wrapper.get("arrayValue", val_wrapper.get("array_value", {}))
            if isinstance(arr, dict):
                values = arr.get("values", [])
            elif isinstance(arr, list):
                values = arr
            else:
                values = []
            return [OTelAttributeParser._unwrap_otel_value(v) for v in values]

        if "kvlistValue" in val_wrapper or "kvlist_value" in val_wrapper:
            kvlist = val_wrapper.get("kvlistValue", val_wrapper.get("kvlist_value", {}))
            if isinstance(kvlist, dict):
                values = kvlist.get("values", [])
            elif isinstance(kvlist, list):
                values = kvlist
            else:
                values = []
            return OTelAttributeParser.parse_attributes(values)

        return val_wrapper
