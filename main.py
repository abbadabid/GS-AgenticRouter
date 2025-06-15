from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant"
)

class State(TypedDict):
    messages : Annotated[list, add_messages]
    message_type : str | None
    next_node: str | None 

class MessageClassifier(BaseModel):
    message_type: Literal["Business", "HR", "Support"] = Field( # Corrected: "Business"
        ...,
        description = "Classify the message into one of the following categories: Business, HR, or Support."
    )

def classify_message(state: State):
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)
    result = classifier_llm.invoke([
        {
            "role" : "system",
            "content" : """Classify the user message as either:
             - 'Business' : if it asks any question related to business operations, sales, or marketing.
             - 'HR' : if it asks any question related to human resources, employee relations, or company policies.
             - 'Support' : if it asks any question related to technical support, product issues, or customer service."""
        },
        {
            "role" : "user",
            "content" : last_message.content
        }
    ])
    return {
        "message_type": result.message_type
    }


def router(state : State):
    message_type = state.get("message_type", "support")
    if message_type == "Business": 
        return {"next_node" : "business_node"}
    elif message_type == "HR" :
        return {"next_node" : "hr_node"}
    else:
        return {"next_node" : "support_node"}

def business_agent(state: State):
    last_message = state["messages"][-1]
    messages = [
        {
            "role" : "system",
            "content" : """You are a business agent. Your task is to assist the user with business-related queries, such as sales, marketing, and operations. Provide detailed and professional responses."""
        },
        {
            "role" : "user",
            "content" : last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {
        "messages" : [HumanMessage(content=reply.content)],
    }

def hr_agent(state: State):
    last_message = state["messages"][-1]
    messages = [
        {
            "role" : "system",
            "content" : """You are an HR agent. Your task is to assist the user with human resources-related queries, such as employee relations, company policies, and benefits. Provide detailed and professional responses."""
        },
        {
            "role" : "user",
            "content" : last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {
        "messages" : [HumanMessage(content=reply.content)],
    }

def support_agent(state: State):
    last_message = state["messages"][-1]
    messages = [
        {
            "role" : "system",
            "content" : """You are a support agent. Your task is to assist the user with technical support, product issues, and customer service queries. Provide detailed and professional responses."""
        },
        {
            "role" : "user",
            "content" : last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {
        "messages" : [HumanMessage(content=reply.content)],
    }

graph_builder = StateGraph(State)

graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("business", business_agent)
graph_builder.add_node("hr", hr_agent)
graph_builder.add_node("support", support_agent)

graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")

graph_builder.add_conditional_edges(
    "router",
    lambda state : state.get("next_node"),
    path_map = {
        "business_node" : "business", 
        "hr_node": "hr",             
        "support_node": "support"  
    }
)
graph_builder.add_edge("business", END)
graph_builder.add_edge("hr", END)
graph_builder.add_edge("support", END)

graph = graph_builder.compile()

def run_chatbot():
    state = {"messages" : [], "message_type" : None, "next_node": None}
    while True:
       user_input = input("Enter a message or 'exit' to quit: ")
       if user_input.lower() == 'exit':
           break
       state["messages"].append(HumanMessage(content=user_input, role="user"))
       final_state = graph.invoke(state)


       if final_state.get("messages") and len(final_state["messages"]) > 0:
           last_message = final_state["messages"][-1]
           print(f"Assistant: {last_message.content}")
       else:
              print("Assistant: No response generated.")

if __name__ == "__main__":
    run_chatbot()
           