from typing import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict):
    approval_status: str


def human_approval(state: State):
    # Pause graph and wait for human response
    approval = interrupt(
        {
            "message": "Do you approve this request?",
            "options": ["accept", "reject"],
        }
    )

    return {"approval_status": approval}


def process_decision(state: State):
    if state["approval_status"] == "accept":
        print("Success")
    else:
        print("Rejected")

    return state


builder = StateGraph(State)

builder.add_node("human_approval", human_approval)
builder.add_node("process_decision", process_decision)

builder.set_entry_point("human_approval")
builder.add_edge("human_approval", "process_decision")
builder.add_edge("process_decision", END)

graph = builder.compile(checkpointer=MemorySaver())

print(graph)


config = {"configurable": {"thread_id": "1"}}

result = graph.invoke({}, config=config)

print('1st attempt')
print(result)
print('----------------------------')

from langgraph.types import Command

result2 = graph.invoke(
    Command(resume="accept"),
    input="",
    config=config
)

print('----------------------------')
print('2nd attempt')
print(result2)
print('----------------------------')
