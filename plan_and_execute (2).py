import os
import asyncio
from langchain import hub
from langchain_openai import ChatOpenAI

from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults

import operator
from typing import Annotated, List, Tuple, TypedDict, Union, Literal

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph

os.environ['OPENAI_API_KEY'] = "..................................."
os.environ['TAVILY_API_KEY'] = ".................................."

tools = [TavilySearchResults(max_results=3)]

prompt = hub.pull("wfh/react-agent-executor")

llm = ChatOpenAI(model="gpt-4-turbo-preview")
agent_executor = create_react_agent(llm, tools, messages_modifier=prompt)

async def plan_and_execute(input_prompt="what is the hometown of the 2024 Australia open winner?"):
    inputs = {"input": input_prompt}
    class PlanExecute(TypedDict):
        input: str
        plan: List[str]
        past_steps: Annotated[List[Tuple], operator.add]
        response: str
    
    class Plan(BaseModel):
        """Plan to follow in future"""
    
        steps: List[str] = Field(
            description="different steps to follow, should be in sorted order"
        )
    
    planner_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """For the given objective, come up with a simple step by step plan. \
    This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
    The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
            ),
            ("placeholder", "{messages}"),
        ]
    )
    planner = planner_prompt | ChatOpenAI(
        model="gpt-4o", temperature=0
    ).with_structured_output(Plan)
    
    class Response(BaseModel):
        """Response to user."""
    
        response: str
    
    
    class Act(BaseModel):
        """Action to perform."""
    
        action: Union[Response, Plan] = Field(
            description="Action to perform. If you want to respond to user, use Response. "
            "If you need to further use tools to get the answer, use Plan."
        )
    
    
    replanner_prompt = ChatPromptTemplate.from_template(
        """For the given objective, come up with a simple step by step plan. \
    This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
    The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.
    
    Your objective was this:
    {input}
    
    Your original plan was this:
    {plan}
    
    You have currently done the follow steps:
    {past_steps}
    
    Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
    )
    
    
    replanner = replanner_prompt | ChatOpenAI(
        model="gpt-4o", temperature=0
    ).with_structured_output(Act)
    
    async def execute_step(state: PlanExecute):
        plan = state["plan"]
        plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        task = plan[0]
        task_formatted = f"""For the following plan:
    {plan_str}\n\nYou are tasked with executing step {1}, {task}."""
        agent_response = await agent_executor.ainvoke(
            {"messages": [("user", task_formatted)]}
        )
        return {
            "past_steps": (task, agent_response["messages"][-1].content),
        }
    
    
    async def plan_step(state: PlanExecute):
        plan = await planner.ainvoke({"messages": [("user", state["input"])]})
        return {"plan": plan.steps}
    
    
    async def replan_step(state: PlanExecute):
        output = await replanner.ainvoke(state)
        if isinstance(output.action, Response):
            return {"response": output.action.response}
        else:
            return {"plan": output.action.steps}
    
    
    def should_end(state: PlanExecute) -> Literal["agent", "__end__"]:
        if "response" in state and state["response"]:
            return "__end__"
        else:
            return "agent"
    
    workflow = StateGraph(PlanExecute)
    
    # Add the plan node
    workflow.add_node("planner", plan_step)
    
    # Add the execution step
    workflow.add_node("agent", execute_step)
    
    # Add a replan node
    workflow.add_node("replan", replan_step)
    
    workflow.set_entry_point("planner")
    
    # From plan we go to agent
    workflow.add_edge("planner", "agent")
    
    # From agent, we replan
    workflow.add_edge("agent", "replan")
    
    workflow.add_conditional_edges(
        "replan",
        should_end,
    )
    
    app = workflow.compile()
    
    config = {"recursion_limit": 50}
    final_output = ""
    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(v)
                final_output = v

    return final_output

async def main():
    input_prompt = input("Enter the input prompt from the user: ")
    output = await plan_and_execute(input_prompt)
    print(f"Final output: {output}")

asyncio.run(main())
