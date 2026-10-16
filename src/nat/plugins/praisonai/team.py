"""``praisonai_team`` workflow/function: a PraisonAI AgentTeam declared in NAT YAML."""
from typing import Literal

from pydantic import BaseModel
from pydantic import Field

from nat.plugin_api import Builder
from nat.plugin_api import Context
from nat.plugin_api import FunctionBaseConfig
from nat.plugin_api import FunctionInfo
from nat.plugin_api import FunctionRef
from nat.plugin_api import LLMRef
from nat.plugin_api import register_function

from . import FRAMEWORK
from . import profiler


class AgentSpec(BaseModel):
    name: str
    role: str | None = None
    goal: str | None = None
    backstory: str | None = None
    instructions: str | None = None
    llm_name: LLMRef | None = Field(default=None, description="Overrides the team-level llm_name.")
    tool_names: list[FunctionRef] = Field(default_factory=list)


class TaskSpec(BaseModel):
    name: str
    description: str
    expected_output: str = "A complete answer."
    agent: str
    context: list[str] = Field(default_factory=list)


class PraisonAITeamConfig(FunctionBaseConfig, name="praisonai_team"):
    """Run a PraisonAI AgentTeam; the workflow input is passed as the team's content."""
    llm_name: LLMRef
    agents: list[AgentSpec]
    tasks: list[TaskSpec] = Field(default_factory=list)
    process: Literal["sequential", "hierarchical"] = "sequential"


@register_function(config_type=PraisonAITeamConfig)
async def praisonai_team(config: PraisonAITeamConfig, builder: Builder):
    from praisonaiagents import Agent, AgentTeam, Task

    profiler.install_token_sink()
    llm_specs = {config.llm_name: await builder.get_llm(config.llm_name, wrapper_type=FRAMEWORK)}
    tools = {}
    for spec in config.agents:
        if spec.llm_name and spec.llm_name not in llm_specs:
            llm_specs[spec.llm_name] = await builder.get_llm(spec.llm_name, wrapper_type=FRAMEWORK)
        tools[spec.name] = await builder.get_tools(spec.tool_names, wrapper_type=FRAMEWORK) if spec.tool_names else []

    async def _run(input_message: str) -> str:
        # Fresh agents per request: PraisonAI agents hold chat history and AgentTeam refuses re-entrancy.
        agents = {
            s.name: Agent(name=s.name, role=s.role, goal=s.goal, backstory=s.backstory, instructions=s.instructions,
                          llm=dict(llm_specs[s.llm_name or config.llm_name]), tools=list(tools[s.name]),
                          hooks=profiler.tool_hooks())
            for s in config.agents
        }
        tasks = {}
        for t in config.tasks:
            tasks[t.name] = Task(name=t.name, description=t.description, expected_output=t.expected_output,
                                 agent=agents[t.agent], context=[tasks[c] for c in t.context])
        team = AgentTeam(agents=list(agents.values()), tasks=list(tasks.values()) or None, process=config.process)
        token = profiler._ACTIVE.set(Context.get().intermediate_step_manager)
        try:
            result = await team.astart(input_message)
        finally:
            profiler._ACTIVE.reset(token)
        # AgentTeam.astart reports task failures as a dict instead of raising; surface them to NAT.
        if isinstance(result, dict) and "task_status" in result:
            failed = [k for k, v in result["task_status"].items() if v == "failed"]
            if failed:
                raise RuntimeError(f"PraisonAI tasks failed: {failed}")
        return str(result)

    yield FunctionInfo.from_fn(_run, description="Run a PraisonAI agent team on the input.")
