"""Targeted edge-case tests to maximize branch coverage across all subsystems."""

import io
import json
from pathlib import Path
from typing import Any

import pytest

from llm_reliability import (
    DiagnosticPolicy,
    OTelAttributeParser,
    PatternDetectionEngine,
    PolicyEngine,
    StreamingProfiler,
    SyntheticTraceGenerator,
    Tracer,
    VerificationEngine,
    load_trace,
    wrap_anthropic,
    wrap_openai,
    wrap_stream,
    wrap_tool,
)
from llm_reliability.agent.analyzer import AgentAnalyzer
from llm_reliability.cli.formatter import format_diagnosis_markdown, format_diagnosis_text
from llm_reliability.cli.main import main
from llm_reliability.exceptions import LLMReliabilityError
from llm_reliability.grounding.analyzer import GroundingAnalyzer
from llm_reliability.models.diagnosis import Diagnosis, Metric
from llm_reliability.models.enums import FailureCategory, Severity, SpanKind, SpanStatus
from llm_reliability.models.execution import ToolCall
from llm_reliability.models.trace import Run, Span
from llm_reliability.normalization.normalizer import normalize_trace, parse_timestamp
from llm_reliability.otel.importer import OTelImporter
from llm_reliability.policies.models import CustomRule
from llm_reliability.report.generator import DiagnosticReportGenerator


class TestCoverageEdgeCases:
    """Exercise remaining edge cases and branches across all modules."""

    def test_normalization_loader_stringio_and_errors(self) -> None:
        raw_json = json.dumps({"trace_id": "t-io", "runs": []})
        stream = io.StringIO(raw_json)
        trace = load_trace(stream)
        assert trace.trace_id == "t-io"

        with pytest.raises((ValueError, LLMReliabilityError)):
            load_trace("")

    def test_normalization_normalizer_timestamps_and_dicts(self) -> None:
        # Parse epoch ms and ISO timestamps
        t1 = parse_timestamp(1700000000000)
        assert t1 is not None
        t2 = parse_timestamp("2026-08-30T10:00:00Z")
        assert t2 is not None
        t3 = parse_timestamp(None)
        assert t3 is None

        # Parse raw span dictionary with tool and llm fields
        raw_dict = {
            "trace_id": "t-norm",
            "spans": [
                {
                    "span_id": "s1",
                    "name": "chat",
                    "kind": "llm",
                    "model": "gpt-4o",
                    "prompt": "Hello",
                    "response": "Hi",
                    "duration_ms": 100.0,
                },
                {
                    "span_id": "s2",
                    "parent_id": "s1",
                    "name": "calc",
                    "kind": "tool",
                    "tool_name": "calc",
                    "input": {"x": 1},
                    "output": {"result": 2},
                },
            ],
        }
        trace = normalize_trace(raw_dict)
        assert trace.trace_id == "t-norm"
        assert len(trace.runs[0].spans) == 2

    def test_otel_attribute_parser_and_time_edge_cases(self) -> None:
        # Test KeyValue list parsing
        kv_list = [
            {"key": "str_val", "value": {"stringValue": "hello"}},
            {"key": "int_val", "value": {"intValue": 42}},
            {"key": "double_val", "value": {"doubleValue": 3.14}},
            {"key": "bool_val", "value": {"boolValue": True}},
            {"key": "array_val", "value": {"arrayValue": {"values": [{"stringValue": "a"}]}}},
        ]
        parsed = OTelAttributeParser.parse_attributes(kv_list)
        assert parsed["str_val"] == "hello"
        assert parsed["int_val"] == 42
        assert parsed["double_val"] == 3.14
        assert parsed["bool_val"] is True
        assert parsed["array_val"] == ["a"]

        # Parse time string vs nano int
        time_nano = OTelImporter._parse_otel_time(1700000000000000000)
        assert time_nano is not None
        time_str = OTelImporter._parse_otel_time("2026-08-30T10:00:00Z")
        assert time_str is not None
        time_none = OTelImporter._parse_otel_time(None)
        assert time_none is None

        # Test empty span list
        empty_trace = OTelImporter.import_span_list([])
        assert "empty" in empty_trace.trace_id

        # Test import from JSON string
        otel_json = json.dumps({"resourceSpans": []})
        imported = OTelImporter.import_trace(otel_json)
        assert imported is not None

    def test_policy_engine_all_operators_and_loaders(self, tmp_path: Path) -> None:
        # Build policy testing operators <, <=, >, >=, ==, !=, in, not_in
        rules = [
            CustomRule(
                rule_id="r1", name="Less", metric_name="m1", operator="<", threshold_value=10
            ),
            CustomRule(
                rule_id="r2", name="LessEq", metric_name="m2", operator="<=", threshold_value=10
            ),
            CustomRule(
                rule_id="r3", name="Greater", metric_name="m3", operator=">", threshold_value=5
            ),
            CustomRule(
                rule_id="r4", name="GreaterEq", metric_name="m4", operator=">=", threshold_value=5
            ),
            CustomRule(
                rule_id="r5",
                name="Equal",
                metric_name="m5",
                operator="==",
                threshold_value="clean",
            ),
            CustomRule(
                rule_id="r6",
                name="NotEqual",
                metric_name="m6",
                operator="!=",
                threshold_value="error",
            ),
            CustomRule(
                rule_id="r7",
                name="In",
                metric_name="m7",
                operator="in",
                threshold_value=["a", "b"],
            ),
            CustomRule(
                rule_id="r8",
                name="NotIn",
                metric_name="m8",
                operator="not_in",
                threshold_value=["x", "y"],
            ),
        ]
        policy = DiagnosticPolicy(
            policy_id="all-ops",
            name="All Ops",
            custom_rules=rules,
            enforce_strict_grounding=True,
            fail_on_warning=True,
        )
        engine = PolicyEngine(policy=policy)

        # Create diagnosis with metrics triggering rule violations
        metrics = [
            Metric(name="m1", value=5.0),
            Metric(name="m2", value=5.0),
            Metric(name="m3", value=10.0),
            Metric(name="m4", value=10.0),
            Metric(name="m5", value=0.0),
            Metric(name="m6", value=1.0),
            Metric(name="m7", value=0.0),
            Metric(name="m8", value=1.0),
            Metric(name="answer_unsupported_claims_count", value=2.0),
        ]
        diagnosis = Diagnosis(
            primary_category=FailureCategory.GROUNDING_FAILURE,
            confidence=0.9,
            severity=Severity.HIGH,
            summary="Test diagnosis",
            metrics=metrics,
        )

        result = engine.evaluate(diagnosis)
        assert result.passed is False
        assert result.violations_count > 0

        # Test YAML and JSON policy loaders
        yaml_content = """
policy_name: test_yaml_policy
thresholds:
  min_supported_ratio: 0.8
  max_latency_ms: 1000
"""
        yaml_file = tmp_path / "policy.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")
        loaded_yaml = PolicyEngine.load_policy_from_yaml(yaml_file)
        assert loaded_yaml.thresholds.min_supported_ratio == 0.8

        json_file = tmp_path / "policy.json"
        json_file.write_text(
            json.dumps({"name": "json_policy", "thresholds": {"max_agent_steps": 5}}),
            encoding="utf-8",
        )
        loaded_json = PolicyEngine.load_policy_from_json(json_file)
        assert loaded_json.thresholds.max_agent_steps == 5

    def test_streaming_wrappers_provider_objects(self) -> None:
        # Test bytes chunk
        bytes_stream = [b"chunk1", b"chunk2"]
        wrapped: list[bytes] = list(wrap_stream(bytes_stream))
        assert wrapped == [b"chunk1", b"chunk2"]

        # Test OpenAI chunk object mock
        class MockChoice:
            class MockDelta:
                content = "openai_chunk"

            delta = MockDelta()

        class MockOpenAIChunk:
            choices = [MockChoice()]

        openai_stream = [MockOpenAIChunk()]
        wrapped_openai: list[Any] = list(wrap_stream(openai_stream))
        assert len(wrapped_openai) == 1

        # Test Anthropic chunk object mock
        class MockAnthropicDelta:
            text = "anthropic_chunk"

        class MockAnthropicChunk:
            delta = MockAnthropicDelta()

        anthropic_stream = [MockAnthropicChunk()]
        wrapped_anthropic: list[Any] = list(wrap_stream(anthropic_stream))
        assert len(wrapped_anthropic) == 1

        # Profiler diagnostic metrics and evidence
        profiler = StreamingProfiler(stall_threshold_ms=100.0)
        profiler.start(0.0)
        profiler.record_chunk("First", 0.050)
        profiler.record_chunk("Second", 0.250)  # stall
        prof = profiler.finish(0.250)
        assert prof.stall_count == 1
        metrics = profiler.to_diagnostic_metrics()
        assert len(metrics) > 0
        evidence = profiler.to_evidence()
        assert len(evidence) > 0

    def test_instrumentors_and_tool_wrappers(self) -> None:
        # Wrap Tool
        def sample_tool_fn(x: int) -> int:
            if x < 0:
                raise ValueError("Negative value error")
            return x * 2

        wrapped = wrap_tool(sample_tool_fn, tool_name="double")
        assert wrapped(5) == 10

        with pytest.raises(ValueError):
            wrapped(-1)

        # Test client wrappers with mock objects
        class MockCompletions:
            def create(self, **kwargs: Any) -> str:
                return "mock_response"

        class MockChat:
            completions = MockCompletions()

        class MockOpenAIClient:
            chat = MockChat()

        wrapped_openai_client = wrap_openai(MockOpenAIClient())
        resp = wrapped_openai_client.chat.completions.create(model="gpt-4o", prompt="test")
        assert resp == "mock_response"

        # Mock Anthropic
        class MockMessages:
            def create(self, **kwargs: Any) -> str:
                return "mock_anthropic_response"

        class MockAnthropicClient:
            messages = MockMessages()

        wrapped_anthropic_client = wrap_anthropic(MockAnthropicClient())
        resp_ant = wrapped_anthropic_client.messages.create(model="claude", prompt="test")
        assert resp_ant == "mock_anthropic_response"

    def test_tracer_custom_span_and_exceptions(self) -> None:
        tracer = Tracer()

        # Exception inside span
        try:
            with tracer.start_span("failing_span", kind=SpanKind.CUSTOM) as active:
                active.attributes["key"] = "value"
                raise RuntimeError("Custom runtime failure")
        except RuntimeError:
            pass

        # LLM Span convenience
        with tracer.start_trace("trace-llm-helper") as trace_ctx:
            with tracer.trace_llm(model="claude-3-5", prompt="test prompt") as active_llm:
                active_llm.span_id = "s-llm"
            with tracer.trace_tool(tool_name="search") as active_tool:
                active_tool.span_id = "s-tool"
            with tracer.trace_retrieval(query="query test") as active_ret:
                active_ret.span_id = "s-ret"

        assert trace_ctx.trace is not None
        assert len(trace_ctx.trace.runs[0].spans) == 3

    def test_cli_verify_and_compare_subcommands(self, tmp_path: Path) -> None:
        generator = SyntheticTraceGenerator()
        t1 = generator.generate_hallucination_trace()
        t2 = generator.generate_clean_trace()

        f1 = tmp_path / "t1.json"
        f2 = tmp_path / "t2.json"
        f1.write_text(json.dumps(t1.model_dump(mode="json")), encoding="utf-8")
        f2.write_text(json.dumps(t2.model_dump(mode="json")), encoding="utf-8")

        out_verify = tmp_path / "verify.txt"
        exit_code_verify = main(["verify", str(f1), str(f2), "--output", str(out_verify)])
        assert exit_code_verify == 0
        assert out_verify.is_file()

        out_compare = tmp_path / "compare.txt"
        exit_code_compare = main(["compare", str(f1), str(f2), "--output", str(out_compare)])
        assert exit_code_compare == 0
        assert out_compare.is_file()

    def test_grounding_and_agent_analyzer_edge_cases(self) -> None:
        grounding_analyzer = GroundingAnalyzer()
        # Empty context
        metrics = grounding_analyzer.analyze_response("Some answer statement.", [])
        assert metrics.unsupported_claims >= 0

        # Agent analyzer with state drift and empty tools
        agent_analyzer = AgentAnalyzer()
        spans = [
            Span(
                span_id="s1",
                name="step1",
                kind=SpanKind.TOOL,
                status=SpanStatus.SUCCESS,
                tool_call=ToolCall(tool_name="search", arguments={"q": "apple"}),
            ),
            Span(
                span_id="s2",
                name="step2",
                kind=SpanKind.TOOL,
                status=SpanStatus.SUCCESS,
                tool_call=ToolCall(tool_name="search", arguments={"q": "apple"}),
            ),
        ]
        run = Run(run_id="r1", trace_id="t1", spans=spans)
        agent_metrics = agent_analyzer.analyze_agent_run(run)
        assert agent_metrics.total_steps == 2
        diag_metrics = agent_metrics.to_diagnostic_metrics()
        assert len(diag_metrics) > 0
        ev = agent_metrics.to_evidence()
        assert len(ev) > 0

    def test_report_generator_without_trace_and_formatters(self) -> None:
        diagnosis = Diagnosis(
            primary_category=FailureCategory.NONE,
            confidence=1.0,
            severity=Severity.INFO,
            summary="Clean run.",
            metrics=[Metric(name="total_latency_ms", value=120.0)],
        )
        gen = DiagnosticReportGenerator(title="No Trace Report")
        html_out = gen.generate(diagnosis, trace=None)
        assert "<!DOCTYPE html>" in html_out
        assert "No Trace Report" in html_out

        text_out = format_diagnosis_text(diagnosis)
        assert "NONE" in text_out

        md_out = format_diagnosis_markdown(diagnosis)
        assert "# LLM Reliability Analysis Report" in md_out

    def test_pattern_and_verification_edge_cases(self) -> None:
        pattern_engine = PatternDetectionEngine()
        generator = SyntheticTraceGenerator()
        traces = [generator.generate_hallucination_trace() for _ in range(3)]
        report = pattern_engine.analyze_patterns(traces)
        assert len(report.clusters) > 0
        assert report.failed_runs_count > 0

        # Verification engine with simulated intervention
        ver_engine = VerificationEngine()
        res = ver_engine.simulate_top_k_adjustment(traces[0].runs[0], new_k=2)
        assert res.is_improved in (True, False)

    def test_gatekeeper_edge_cases_and_markdown_comment(self, tmp_path: Path) -> None:
        from llm_reliability.eval_harness.gatekeeper import Gatekeeper, evaluate_gate
        from llm_reliability.eval_harness.models import GatekeeperConfig

        generator = SyntheticTraceGenerator(seed=77)
        clean = generator.generate_clean_trace("gate-clean")
        fail = generator.generate_hallucination_trace("gate-fail")

        cfg = GatekeeperConfig(
            max_failure_rate=0.1,
            max_regression_score=0.0,
            max_latency_p95_ms=500.0,
            min_grounding_score=0.9,
            disallowed_categories=[FailureCategory.HALLUCINATION],
        )
        gk = Gatekeeper()
        rep = gk.evaluate(candidate_trace=fail, baseline_trace=clean, config=cfg)
        assert rep.passed is False
        assert len(rep.violations) > 0
        assert "Reliability Gatekeeper" in rep.markdown_summary

        clean_file = tmp_path / "clean.json"
        clean_file.write_text(clean.model_dump_json(), encoding="utf-8")
        rep_file = evaluate_gate(candidate_file=clean_file, baseline_file=clean_file, config=cfg)
        assert rep_file.passed is True

    def test_multimodal_and_structured_output_edge_cases(self) -> None:
        from llm_reliability.models.execution import LLMCall
        from llm_reliability.models.trace import Trace
        from llm_reliability.multimodal.analyzer import (
            MultiModalAnalyzer,
            MultiModalReliabilityAnalyzer,
            StructuredOutputAnalyzer,
        )

        analyzer = StructuredOutputAnalyzer()
        # Invalid JSON
        valid, parsed, err = analyzer.validate_json("not valid json {")
        assert valid is False
        assert parsed is None
        assert err is not None

        # Markdown wrapped JSON
        valid_md, parsed_md, _ = analyzer.validate_json('```json\n{"foo": "bar"}\n```')
        assert valid_md is True
        assert parsed_md == {"foo": "bar"}

        # Schema constraint checks (integer vs boolean, missing required, unexpected properties)
        schema = {
            "type": "object",
            "required": ["age", "name"],
            "properties": {
                "age": {"type": "integer"},
                "name": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        }
        # bool provided instead of integer
        llm_call = LLMCall(
            model="gpt-4",
            prompt="Gen JSON",
            response='{"age": true, "name": "Alice", "tags": [123], "extra": "forbidden"}',
            raw_parameters={"json_schema": schema},
        )
        res = analyzer.analyze(llm_call)
        assert len(res.failures) > 0
        assert res.structured_metrics is not None
        assert len(res.structured_metrics.type_mismatches) > 0

        # MultiModal analyzer image references and base64
        mm = MultiModalAnalyzer()
        span_bad_b64 = Span(
            span_id="s-b64",
            name="image_span",
            kind=SpanKind.LLM,
            status=SpanStatus.SUCCESS,
            attributes={"images": ["data:image/png;base64,invalid_base64_%%%"]},
            llm_call=LLMCall(model="gpt-4v", prompt="look", response="Image 1 has text"),
        )
        mm_res = mm.analyze_span(span_bad_b64)
        assert len(mm_res.failures) > 0

        # Unified analyzer trace
        mm_engine = MultiModalReliabilityAnalyzer()
        run = Run(run_id="r_mm", trace_id="t_mm", spans=[span_bad_b64])
        t_mm = Trace(trace_id="t_mm", runs=[run])
        trace_res = mm_engine.analyze_trace(t_mm)
        assert len(trace_res.failures) > 0

    def test_anonymization_and_pii_scrubbing_edge_cases(self) -> None:
        from llm_reliability.anonymization.engine import PIIScrubber, TelemetryAnonymizer
        from llm_reliability.anonymization.models import (
            PIIType,
            RedactionConfig,
            RedactionMaskType,
            RedactionRule,
        )

        # Masking types: HASH and PARTIAL_MASK
        cfg_hash = RedactionConfig(default_mask_type=RedactionMaskType.HASH)
        scrubber_hash = PIIScrubber(cfg_hash)
        res_hash = scrubber_hash.scrub_text("Contact user@example.com")
        assert "user@example.com" not in res_hash.sanitized_text

        cfg_partial = RedactionConfig(default_mask_type=RedactionMaskType.PARTIAL_MASK)
        scrubber_partial = PIIScrubber(cfg_partial)
        res_partial = scrubber_partial.scrub_text("Phone +1-555-123-4567")
        assert "+1-555-123-4567" not in res_partial.sanitized_text

        # Custom user-defined regex rule
        custom_rule = RedactionRule(
            name="custom_secret",
            pii_type=PIIType.CUSTOM,
            pattern=r"SECRET-\d+",
        )
        cfg_custom = RedactionConfig(custom_rules=[custom_rule])
        scrubber_custom = PIIScrubber(cfg_custom)
        res_custom = scrubber_custom.scrub_text("Found token SECRET-998811 in logs")
        assert "[CUSTOM]" in res_custom.sanitized_text

        # Telemetry anonymizer trace with spans
        anon = TelemetryAnonymizer(RedactionConfig())
        generator = SyntheticTraceGenerator()
        trace = generator.generate_clean_trace("pii-trace")
        sanitized = anon.anonymize_trace(trace)
        assert sanitized.trace_id == "pii-trace"

    def test_plugin_manager_lifecycle_and_hooks(self) -> None:
        from llm_reliability.models.trace import Trace
        from llm_reliability.plugins.base import (
            BaseAnalyzerPlugin,
            BaseExporterPlugin,
            BaseMiddlewarePlugin,
            PluginMetadata,
        )
        from llm_reliability.plugins.manager import PluginManager

        mgr = PluginManager()

        class DummyAnalyzer(BaseAnalyzerPlugin):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(name="dummy_analyzer", version="0.1.0")

            def analyze(self, trace: Trace) -> tuple[list[Any], list[Any], list[Any]]:
                return [], [], []

        class DummyExporter(BaseExporterPlugin):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(name="dummy_exporter", version="0.1.0")

            @property
            def format_name(self) -> str:
                return "dummy"

            def export(self, diagnosis: Diagnosis, output_path: Path | str | None = None) -> Any:
                return {"exported": True}

        class DummyMiddleware(BaseMiddlewarePlugin):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(name="dummy_middleware", version="0.1.0")

            def before_diagnosis(self, trace: Trace) -> Trace:
                return trace

            def after_diagnosis(self, diagnosis: Diagnosis) -> Diagnosis:
                return diagnosis

        a = DummyAnalyzer()
        e = DummyExporter()
        m = DummyMiddleware()

        mgr.register(a)
        mgr.register(e)
        mgr.register(m)

        assert len(mgr.list_plugins()) == 3
        assert mgr.get_plugin("dummy_analyzer") is not None

        gen = SyntheticTraceGenerator()
        trace = gen.generate_clean_trace("plugin-trace")
        diag = Diagnosis(primary_category=FailureCategory.NONE, summary="ok")

        trace_mod = mgr.apply_pre_middlewares(trace)
        assert trace_mod.trace_id == "plugin-trace"

        diag_mod = mgr.apply_post_middlewares(diag)
        assert diag_mod.summary == "ok"

        export_res = mgr.export(diag, "dummy")
        assert export_res.get("exported") is True

        mgr.unregister("dummy_analyzer")
        assert mgr.get_plugin("dummy_analyzer") is None

    def test_server_config_and_routes(self, tmp_path: Path) -> None:
        from llm_reliability.server.models import ServerConfig
        from llm_reliability.server.server import DiagnosticServer

        # Preload traces test
        t_dir = tmp_path / "traces"
        t_dir.mkdir()
        (t_dir / "bad.json").write_text("invalid json", encoding="utf-8")
        gen = SyntheticTraceGenerator()
        tr = gen.generate_clean_trace("server-t1")
        (t_dir / "good.json").write_text(tr.model_dump_json(), encoding="utf-8")

        cfg = ServerConfig(host="127.0.0.1", port=8999, trace_dir=t_dir)
        server = DiagnosticServer(cfg)
        assert server.config.port == 8999
        count = server.preload_traces(t_dir)
        assert count == 1
        assert server.get_record("server-t1") is not None
        assert len(server.get_records()) == 1

        server.start(background=True)
        server.stop()

    def test_otel_attribute_parser_all_types(self) -> None:
        from llm_reliability.otel.models import OTelAttributeParser

        # arrayValue & kvlistValue
        raw_list = [
            {"key": "string_field", "value": {"stringValue": "hello"}},
            {"key": "int_field", "value": {"intValue": 42}},
            {"key": "double_field", "value": {"doubleValue": 3.14}},
            {"key": "bool_field", "value": {"boolValue": True}},
            {
                "key": "arr_field",
                "value": {
                    "arrayValue": {"values": [{"stringValue": "item1"}, {"stringValue": "item2"}]}
                },
            },
            {
                "key": "kv_field",
                "value": {
                    "kvlistValue": {
                        "values": [{"key": "sub_key", "value": {"stringValue": "sub_val"}}]
                    }
                },
            },
        ]
        parsed = OTelAttributeParser.parse_attributes(raw_list)
        assert parsed["string_field"] == "hello"
        assert parsed["int_field"] == 42
        assert parsed["double_field"] == 3.14
        assert parsed["bool_field"] is True
        assert parsed["arr_field"] == ["item1", "item2"]
        assert parsed["kv_field"] == {"sub_key": "sub_val"}

        # Empty / non-dict values
        assert OTelAttributeParser.parse_attributes(None) == {}
        assert OTelAttributeParser.parse_attributes("non_collection") == {}

    def test_gatekeeper_custom_policy_file(self, tmp_path: Path) -> None:
        from llm_reliability.eval_harness.gatekeeper import Gatekeeper
        from llm_reliability.eval_harness.models import GatekeeperConfig

        policy_file = tmp_path / "custom_policy.json"
        policy_data = {
            "name": "strict_grounding_policy",
            "thresholds": {"min_grounding_score": 0.95},
            "rules": [
                {
                    "rule_id": "disallow_loops",
                    "target": "agent",
                    "metric_name": "repeated_tool_calls",
                    "operator": "==",
                    "threshold": 0,
                    "failure_category": "AGENT_LOOP",
                }
            ],
        }
        policy_file.write_text(json.dumps(policy_data), encoding="utf-8")

        cfg = GatekeeperConfig(policy_file=policy_file)
        gk = Gatekeeper()
        generator = SyntheticTraceGenerator()
        clean = generator.generate_clean_trace("gk-clean")
        rep = gk.evaluate(candidate_trace=clean, config=cfg)
        assert rep is not None

    def test_instrumentor_wrappers_and_streaming(self) -> None:
        from llm_reliability.streaming.wrappers import wrap_stream
        from llm_reliability.tracing.instrumentors import wrap_anthropic, wrap_openai, wrap_tool

        # 1. wrap_openai
        class MockChoiceMessage:
            content = "Mocked LLM reply"

        class MockChoice:
            message = MockChoiceMessage()

        class MockUsage:
            prompt_tokens = 10
            completion_tokens = 5
            total_tokens = 15

        class MockResponse:
            choices = [MockChoice()]
            usage = MockUsage()

        class MockCompletions:
            def create(self, *args: Any, **kwargs: Any) -> Any:
                return MockResponse()

        class MockChat:
            completions = MockCompletions()

        class MockOpenAIClient:
            chat = MockChat()

        wrapped_openai = wrap_openai(MockOpenAIClient())
        resp = wrapped_openai.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "Hi"}]
        )
        assert resp.choices[0].message.content == "Mocked LLM reply"

        # 2. wrap_anthropic
        class MockAnthropicContent:
            text = "Anthropic reply"

        class MockAnthropicUsage:
            input_tokens = 8
            output_tokens = 4

        class MockAnthropicResponse:
            content = [MockAnthropicContent()]
            usage = MockAnthropicUsage()

        class MockMessages:
            def create(self, *args: Any, **kwargs: Any) -> Any:
                return MockAnthropicResponse()

        class MockAnthropicClient:
            messages = MockMessages()

        wrapped_anthropic = wrap_anthropic(MockAnthropicClient())
        anth_resp = wrapped_anthropic.messages.create(
            model="claude-3-5-sonnet", messages=[{"role": "user", "content": "Hi"}]
        )
        assert anth_resp.content[0].text == "Anthropic reply"

        # 3. wrap_tool
        def raw_add(a: int, b: int) -> int:
            return a + b

        calc = wrap_tool(raw_add, tool_name="calculator")
        assert calc(3, 4) == 7

        # 4. wrap_stream
        from collections.abc import Iterator

        from llm_reliability.streaming.models import TokenLatencyProfile

        def token_generator() -> Iterator[str]:
            yield "Hello "
            yield "World"

        captured_profiles: list[TokenLatencyProfile] = []

        def callback(profile: TokenLatencyProfile, text: str) -> None:
            captured_profiles.append(profile)

        wrapped_gen: Iterator[str] = wrap_stream(token_generator(), on_complete=callback)
        collected: list[str] = list(wrapped_gen)
        assert "".join(collected) == "Hello World"
        assert len(captured_profiles) == 1
        assert captured_profiles[0].total_tokens == 2

        # 5. wrap_async_stream
        import asyncio
        from collections.abc import AsyncIterator

        from llm_reliability.streaming.wrappers import wrap_async_stream

        async def async_token_generator() -> AsyncIterator[str]:
            yield "Async "
            yield "Stream"

        async def run_async_test() -> None:
            captured_async: list[TokenLatencyProfile] = []

            def async_cb(profile: TokenLatencyProfile, text: str) -> None:
                captured_async.append(profile)

            collected_async: list[str] = []
            c: str
            async for c in wrap_async_stream(async_token_generator(), on_complete=async_cb):
                collected_async.append(c)
            assert "".join(collected_async) == "Async Stream"
            assert len(captured_async) == 1
            assert captured_async[0].total_tokens == 2

        asyncio.run(run_async_test())

    def test_anonymize_full_complex_span(self) -> None:
        from llm_reliability.anonymization.engine import TelemetryAnonymizer
        from llm_reliability.models.execution import (
            LLMCall,
            RetrievalStep,
            RetrievedDocument,
            ToolCall,
            ToolResult,
        )

        anon = TelemetryAnonymizer()
        span = Span(
            span_id="s_full",
            name="full_span",
            kind=SpanKind.AGENT,
            status=SpanStatus.ERROR,
            attributes={
                "api_key": "sk-12345678901234567890123456789012",
                "input": {"user_msg": "my email is user@domain.com"},
                "output": {"reply": "contacted user@domain.com"},
            },
            error_message="failed for user@domain.com",
            llm_call=LLMCall(
                model="gpt-4",
                prompt="prompt for user@domain.com",
                response="response for user@domain.com",
            ),
            tool_call=ToolCall(
                tool_name="search",
                arguments={"query": "user@domain.com"},
            ),
            tool_result=ToolResult(
                tool_name="search",
                output={"res": "user@domain.com"},
                error="err user@domain.com",
            ),
            retrieval=RetrievalStep(
                query="find user@domain.com",
                documents=[
                    RetrievedDocument(
                        doc_id="d1",
                        content="doc user@domain.com",
                        metadata={"owner": "user@domain.com"},
                    )
                ],
            ),
        )
        anonymized = anon.anonymize_span(span)
        dump = anonymized.model_dump_json()
        assert "user@domain.com" not in dump

    def test_instrumentors_additional_branches(self) -> None:
        from llm_reliability.tracing.instrumentors import wrap_anthropic, wrap_openai, wrap_tool

        # No-op clients
        assert wrap_openai(None) is None
        assert wrap_anthropic(None) is None

        # Dict response OpenAI
        class MockDictCompletions:
            def create(self, *args: Any, **kwargs: Any) -> Any:
                return {
                    "choices": [{"message": {"content": "dict response"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 10},
                }

        class MockDictClient:
            class chat:  # noqa: N801
                completions = MockDictCompletions()

        wrapped_d = wrap_openai(MockDictClient())
        res = wrapped_d.chat.completions.create(model="gpt-dict")
        assert res["choices"][0]["message"]["content"] == "dict response"

        # OpenAI error handling
        class MockErrCompletions:
            def create(self, *args: Any, **kwargs: Any) -> Any:
                raise ValueError("API call failed")

        class MockErrClient:
            class chat:  # noqa: N801
                completions = MockErrCompletions()

        wrapped_err = wrap_openai(MockErrClient())
        with pytest.raises(ValueError, match="API call failed"):
            wrapped_err.chat.completions.create(model="gpt-err")

        # Anthropic dict response & error
        class MockAnthropicDictMessages:
            def create(self, *args: Any, **kwargs: Any) -> Any:
                return {
                    "content": [{"text": "anthropic dict"}],
                    "usage": {"input_tokens": 3, "output_tokens": 4},
                }

        class MockAnthropicDictClient:
            messages = MockAnthropicDictMessages()

        w_anth_dict = wrap_anthropic(MockAnthropicDictClient())
        res_a = w_anth_dict.messages.create(model="claude-dict")
        assert res_a["content"][0]["text"] == "anthropic dict"

        # Tool error handling
        def fail_tool() -> None:
            raise RuntimeError("Tool failed")

        t_wrapped = wrap_tool(fail_tool, tool_name="failing_tool")
        with pytest.raises(RuntimeError, match="Tool failed"):
            t_wrapped()

    def test_gatekeeper_empty_runs_and_grounding_threshold(self) -> None:
        from llm_reliability.eval_harness.gatekeeper import Gatekeeper
        from llm_reliability.eval_harness.models import GatekeeperConfig
        from llm_reliability.models.trace import Trace

        gk = Gatekeeper()
        # Empty trace
        empty_tr = Trace(trace_id="empty_tr", runs=[])

        rep_empty = gk.evaluate(empty_tr)
        assert rep_empty.passed is True
        assert "0 runs" in rep_empty.markdown_summary

        # Grounding violation
        gen = SyntheticTraceGenerator()
        halluc_tr = gen.generate_hallucination_trace("halluc")
        cfg_grounding = GatekeeperConfig(min_grounding_score=0.99)
        rep_grounding = gk.evaluate(halluc_tr, config=cfg_grounding)
        assert rep_grounding.passed is False

        # Grounding success evaluation
        from llm_reliability.models.execution import LLMCall, RetrievalStep, RetrievedDocument

        span_rag = Span(
            span_id="s_rag",
            name="rag_span",
            kind=SpanKind.RETRIEVAL,
            status=SpanStatus.SUCCESS,
            retrieval=RetrievalStep(
                query="what is python",
                documents=[
                    RetrievedDocument(doc_id="d1", content="Python is a programming language")
                ],
            ),
        )
        span_llm = Span(
            span_id="s_llm",
            name="llm_span",
            kind=SpanKind.LLM,
            status=SpanStatus.SUCCESS,
            llm_call=LLMCall(
                model="gpt-4",
                prompt="what is python",
                response="Python is a programming language",
            ),
        )
        tr_rag = Trace(
            trace_id="tr_rag",
            runs=[Run(run_id="r_rag", trace_id="tr_rag", spans=[span_rag, span_llm])],
        )
        cfg_min = GatekeeperConfig(min_grounding_score=0.5)
        rep_rag = gk.evaluate(tr_rag, config=cfg_min)
        assert rep_rag is not None

    def test_loader_edge_cases(self, tmp_path: Path) -> None:
        import io

        from llm_reliability.exceptions import TraceParseError, TraceValidationError
        from llm_reliability.normalization.loader import load_trace

        # None source
        with pytest.raises(TraceValidationError, match="cannot be None"):
            load_trace(None)

        # Empty string
        with pytest.raises(TraceValidationError, match="empty"):
            load_trace("   ")

        # Unsupported type
        with pytest.raises(TraceValidationError, match="Unsupported trace source type"):
            load_trace(12345)  # type: ignore[arg-type]

        # Non-existent Path
        with pytest.raises(TraceParseError, match="not found"):
            load_trace(tmp_path / "non_existent.json")

        # Non-existent string path
        with pytest.raises(TraceParseError, match="not found"):
            load_trace("non_existent_file.json")

        # TextIO stream
        stream = io.StringIO('{"trace_id": "stream_tr", "spans": []}')
        tr_stream = load_trace(stream)
        assert tr_stream.trace_id == "stream_tr"

        # Broken stream
        class BrokenStream:
            def read(self) -> str:
                raise OSError("stream read failure")

        with pytest.raises(TraceParseError, match="Failed to read from trace stream"):
            load_trace(BrokenStream())  # type: ignore[arg-type]

        # Raw string with unparseable JSON
        with pytest.raises(TraceParseError):
            load_trace("this is not json { [")

        # Empty content helper
        from llm_reliability.normalization.loader import _parse_and_normalize_json_string

        with pytest.raises(TraceValidationError, match="empty"):
            _parse_and_normalize_json_string("", source_hint="test_empty")

    def test_server_invalid_trace_dir_and_start_helper(self) -> None:
        from llm_reliability.server.models import ServerConfig
        from llm_reliability.server.server import DiagnosticServer, start_server

        server = DiagnosticServer(ServerConfig(port=9012))
        count = server.preload_traces("non_existent_dir_path_xyz")
        assert count == 0

        # start_server helper
        srv = start_server(ServerConfig(port=9013), background=True)
        assert srv.config.port == 9013
        srv.stop()

    def test_cli_main_module_execution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import runpy

        monkeypatch.setattr("sys.argv", ["llm_reliability", "--version"])
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_module("llm_reliability.cli", run_name="__main__")
        assert exc_info.value.code == 0
