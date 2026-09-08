"""Central Plugin Manager handling registration, entrypoint discovery, and execution."""

import importlib.metadata
import logging
from pathlib import Path
from typing import Any

from llm_reliability.models.diagnosis import Diagnosis, Evidence, Failure, Metric
from llm_reliability.models.trace import Trace
from llm_reliability.plugins.base import (
    BaseAnalyzerPlugin,
    BaseExporterPlugin,
    BaseMiddlewarePlugin,
    BasePlugin,
    PluginMetadata,
)

logger = logging.getLogger(__name__)


class PluginManager:
    """Manages lifecycle, registration, discovery, and isolated execution of plugins."""

    def __init__(self) -> None:
        self._plugins: dict[str, BasePlugin] = {}
        self._analyzers: list[BaseAnalyzerPlugin] = []
        self._exporters: dict[str, BaseExporterPlugin] = {}
        self._middlewares: list[BaseMiddlewarePlugin] = []

    def register(self, plugin: BasePlugin) -> None:
        """Register a plugin instance with lifecycle initialization."""
        name = plugin.metadata.name
        if name in self._plugins:
            self.unregister(name)

        plugin.initialize()
        self._plugins[name] = plugin

        if isinstance(plugin, BaseAnalyzerPlugin):
            self._analyzers.append(plugin)
        if isinstance(plugin, BaseExporterPlugin):
            self._exporters[plugin.format_name.lower()] = plugin
        if isinstance(plugin, BaseMiddlewarePlugin):
            self._middlewares.append(plugin)

    def unregister(self, name: str) -> None:
        """Unload and unregister a plugin by name."""
        plugin = self._plugins.pop(name, None)
        if plugin is not None:
            try:
                plugin.shutdown()
            except Exception as e:
                logger.warning(f"Error during plugin '{name}' shutdown: {e}")

            if isinstance(plugin, BaseAnalyzerPlugin) and plugin in self._analyzers:
                self._analyzers.remove(plugin)
            if (
                isinstance(plugin, BaseExporterPlugin)
                and plugin.format_name.lower() in self._exporters
            ):
                del self._exporters[plugin.format_name.lower()]
            if isinstance(plugin, BaseMiddlewarePlugin) and plugin in self._middlewares:
                self._middlewares.remove(plugin)

    def get_plugin(self, name: str) -> BasePlugin | None:
        """Get plugin instance by name."""
        return self._plugins.get(name)

    def list_plugins(self) -> list[PluginMetadata]:
        """Return list of all registered plugin metadata."""
        return [p.metadata for p in self._plugins.values()]

    def discover_entrypoints(self, group: str = "llm_reliability.plugins") -> int:
        """Discover and load third-party plugins from installed Python package entry points."""
        discovered_count = 0
        try:
            entry_points = importlib.metadata.entry_points()
            if hasattr(entry_points, "select"):
                matches = list(entry_points.select(group=group))
            else:
                raw_matches = getattr(entry_points, "get", lambda _g, _d: _d)(group, [])
                matches = list(raw_matches)

            for ep in matches:
                try:
                    plugin_class = ep.load()
                    plugin_instance = plugin_class()
                    if isinstance(plugin_instance, BasePlugin):
                        self.register(plugin_instance)
                        discovered_count += 1
                except Exception as e:
                    logger.warning(f"Failed to load plugin entry point '{ep.name}': {e}")
        except Exception as e:
            logger.warning(f"Error inspecting entry points for group '{group}': {e}")

        return discovered_count

    def run_analyzers(self, trace: Trace) -> tuple[list[Failure], list[Evidence], list[Metric]]:
        """Execute all registered custom analyzer plugins with error isolation."""
        failures: list[Failure] = []
        evidence: list[Evidence] = []
        metrics: list[Metric] = []

        for analyzer in self._analyzers:
            try:
                result = analyzer.analyze(trace)
                if isinstance(result, tuple):
                    res_failures, res_evidence, res_metrics = result
                    failures.extend(res_failures)
                    evidence.extend(res_evidence)
                    metrics.extend(res_metrics)
                elif isinstance(result, list):
                    failures.extend(result)
            except Exception as e:
                logger.error(
                    f"Error running analyzer plugin '{analyzer.metadata.name}': {e}",
                    exc_info=True,
                )

        return failures, evidence, metrics

    def apply_pre_middlewares(self, trace: Trace) -> Trace:
        """Run all pre-diagnosis middleware with error isolation."""
        current_trace = trace
        for mw in self._middlewares:
            try:
                current_trace = mw.before_diagnosis(current_trace)
            except Exception as e:
                logger.error(
                    f"Error running pre-middleware plugin '{mw.metadata.name}': {e}",
                    exc_info=True,
                )
        return current_trace

    def apply_post_middlewares(self, diagnosis: Diagnosis) -> Diagnosis:
        """Run all post-diagnosis middleware with error isolation."""
        current_diagnosis = diagnosis
        for mw in self._middlewares:
            try:
                current_diagnosis = mw.after_diagnosis(current_diagnosis)
            except Exception as e:
                logger.error(
                    f"Error running post-middleware plugin '{mw.metadata.name}': {e}",
                    exc_info=True,
                )
        return current_diagnosis

    def export(
        self, diagnosis: Diagnosis, format_name: str, output_path: Path | str | None = None
    ) -> Any:
        """Export diagnosis via registered custom exporter plugin."""
        exporter = self._exporters.get(format_name.lower())
        if exporter is None:
            available = list(self._exporters.keys())
            raise ValueError(
                f"No exporter plugin registered for format '{format_name}'. Available: {available}"
            )
        return exporter.export(diagnosis, output_path)

    def clear(self) -> None:
        """Unload and clear all plugins."""
        for name in list(self._plugins.keys()):
            self.unregister(name)


# Global default plugin manager singleton
_GLOBAL_PLUGIN_MANAGER = PluginManager()


def get_plugin_manager() -> PluginManager:
    """Return the global default plugin manager instance."""
    return _GLOBAL_PLUGIN_MANAGER


def register_plugin(plugin: BasePlugin) -> None:
    """Convenience helper to register a plugin globally."""
    _GLOBAL_PLUGIN_MANAGER.register(plugin)
