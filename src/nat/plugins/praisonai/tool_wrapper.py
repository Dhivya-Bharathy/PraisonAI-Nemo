"""Wrap a NAT Function as a PraisonAI BaseTool.

PraisonAI's AgentTeam.astart runs each task in a worker thread (run_in_executor) and drives an
``async`` tool on a *fresh* event loop there. NAT functions may hold resources bound to the
builder's loop, so the tool hops back to that loop, carrying the caller's contextvars so the
NAT FUNCTION_START/END events nest under the active workflow span.
"""
import asyncio
import concurrent.futures
import contextvars
import copy

from praisonaiagents.tools.base import BaseTool

from nat.plugin_api import Builder
from nat.plugin_api import Function
from nat.plugin_api import register_tool_wrapper

from . import FRAMEWORK


def _submit(coro, loop: asyncio.AbstractEventLoop, ctx: contextvars.Context) -> concurrent.futures.Future:
    cf: concurrent.futures.Future = concurrent.futures.Future()

    def _start():
        task = loop.create_task(coro, context=ctx)  # Python >= 3.11

        def _done(t: asyncio.Task):
            if t.cancelled():
                cf.cancel()
            elif t.exception() is not None:
                cf.set_exception(t.exception())
            else:
                cf.set_result(t.result())

        task.add_done_callback(_done)

    loop.call_soon_threadsafe(_start)
    return cf


class NATFunctionTool(BaseTool):

    def __init__(self, name: str, fn: Function, loop: asyncio.AbstractEventLoop):
        self.name = name
        self.description = (fn.description or name).strip()
        schema = copy.deepcopy(fn.input_schema.model_json_schema()) if fn.input_schema else {}
        schema.pop("title", None)
        self.parameters = {"type": "object", "properties": schema.get("properties", {}),
                           "required": schema.get("required", []), **({"$defs": schema["$defs"]} if "$defs" in schema else {})}
        self._fn = fn
        self._loop = loop
        super().__init__()

    async def run(self, **kwargs):  # noqa: D401
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            return await self._fn.acall_invoke(**kwargs)
        cf = _submit(self._fn.acall_invoke(**kwargs), self._loop, contextvars.copy_context())
        if running is None:
            return cf.result()
        return await asyncio.wrap_future(cf)


@register_tool_wrapper(wrapper_type=FRAMEWORK)
def praisonai_tool_wrapper(name: str, fn: Function, _builder: Builder):
    return NATFunctionTool(name, fn, asyncio.get_running_loop())
