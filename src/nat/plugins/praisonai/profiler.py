"""Bridge PraisonAI usage/tool signals into NAT IntermediateStep events.

* LLM: PraisonAI's global TokenCollector calls ``sink.persist`` synchronously, once per provider
  response, on the thread that made the call (verified: contextvars are intact there). We emit an
  LLM_START/LLM_END pair with matching UUID and the token usage. PraisonAI has no per-call
  *start* signal today, so the START is emitted at completion time (latency is not measured).
* Tools: per-agent HookRegistry BEFORE_TOOL/AFTER_TOOL -> TOOL_START/TOOL_END.

Events are only emitted while ``_ACTIVE`` is set (inside a praisonai_team invocation), so other
PraisonAI usage in the same process is untouched and concurrent NAT requests stay separated.
"""
import contextvars
import threading
import time
from uuid import uuid4

from nat.builder.context import Context
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import StreamEventData
from nat.data_models.intermediate_step import TraceMetadata
from nat.data_models.intermediate_step import UsageInfo
from nat.data_models.token_usage import TokenUsageBaseModel

from . import FRAMEWORK

# Holds the request's IntermediateStepManager while a praisonai_team invocation runs. START and END
# must go through the SAME manager instance: Context.get() builds a new Context (and a new manager with
# its own outstanding-start table) on every call, so an END pushed via a fresh manager is dropped.
_ACTIVE: contextvars.ContextVar = contextvars.ContextVar("nat_praisonai_active", default=None)
_install_lock = threading.Lock()
_installed = False


def _meta(**kw) -> TraceMetadata:
    return TraceMetadata(provided_metadata={"framework": FRAMEWORK, **kw})


class NATTokenSink:
    """TokenUsageSinkProtocol implementation; chains to any previously installed sink."""

    def __init__(self, previous=None):
        self._previous = previous

    def persist(self, task_id, agent_name, model, metrics, metadata=None):
        mgr = _ACTIVE.get()
        if mgr is not None:
            usage = TokenUsageBaseModel(prompt_tokens=metrics.input_tokens,
                                        completion_tokens=metrics.output_tokens,
                                        cached_tokens=getattr(metrics, "cached_tokens", 0) or 0,
                                        reasoning_tokens=getattr(metrics, "reasoning_tokens", 0) or 0,
                                        total_tokens=metrics.total_tokens)
            uid = str(uuid4())
            now = time.time()
            mgr.push_intermediate_step(IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_START, name=model, UUID=uid,
                metadata=_meta(agent=agent_name), usage_info=UsageInfo(num_llm_calls=1)))
            mgr.push_intermediate_step(IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_END, name=model, UUID=uid, span_event_timestamp=now,
                data=StreamEventData(), metadata=_meta(agent=agent_name),
                usage_info=UsageInfo(token_usage=usage, num_llm_calls=1)))
        if self._previous is not None:
            self._previous.persist(task_id=task_id, agent_name=agent_name, model=model, metrics=metrics,
                                   metadata=metadata)


def install_token_sink() -> None:
    global _installed
    with _install_lock:
        if _installed:
            return
        from praisonaiagents.telemetry.token_collector import get_token_collector
        collector = get_token_collector()
        collector.set_sink(NATTokenSink(previous=getattr(collector, "_sink", None)))
        _installed = True


def tool_hooks():
    """Return a PraisonAI ``hooks=`` value emitting TOOL_START/TOOL_END."""
    from praisonaiagents.hooks import HookEvent
    from praisonaiagents.hooks import HookRegistry

    reg = HookRegistry()
    open_tools: dict[str, list[str]] = {}  # hooks may run in different contexts; keep pairing state here

    @reg.on(HookEvent.BEFORE_TOOL)
    def _before(data):
        mgr = _ACTIVE.get()
        if mgr is None:
            return None
        uid = str(uuid4())
        open_tools.setdefault(data.tool_name, []).append(uid)
        mgr.push_intermediate_step(IntermediateStepPayload(
            event_type=IntermediateStepType.TOOL_START, name=data.tool_name, UUID=uid,
            data=StreamEventData(input=data.tool_input), metadata=_meta(agent=data.agent_name)))
        return None

    @reg.on(HookEvent.AFTER_TOOL)
    def _after(data):
        mgr = _ACTIVE.get()
        pending = open_tools.get(data.tool_name)
        if mgr is None or not pending:
            return None
        uid = pending.pop(0)
        mgr.push_intermediate_step(IntermediateStepPayload(
            event_type=IntermediateStepType.TOOL_END, name=data.tool_name, UUID=uid,
            span_event_timestamp=time.time(),
            data=StreamEventData(input=data.tool_input, output=str(data.tool_output)),
            metadata=_meta(agent=data.agent_name)))
        return None

    return {"registry": reg}
