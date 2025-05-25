import os
import click
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from typing_extensions import TypedDict
from typing_extensions import Annotated
from typing import Optional, Literal
from langgraph.graph import START, StateGraph

load_dotenv()


# Optional: Set up LangSmith tracing keys (commented out)
# if not os.environ.get("LANGSMITH_API_KEY"):
#     os.environ["LANGSMITH_API_KEY"] = ''
#     os.environ["LANGSMITH_TRACING"] = "true"

# Define the structure of the state object passed between graph nodes
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str
    check: str


class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]


class Mark(TypedDict):
    """Bool response."""
    result: Annotated[Optional[bool], None, "Does the sql query match the question?"]


# User prompt for asking questions
user_prompt = "Question: {input}"
# System instructions to guide the model on how to construct a SQL query
system_message = """
Given an input question, create a syntactically correct {dialect} query to
run to help find the answer. Unless the user specifies in his question a
specific number of examples they wish to obtain. You can order the results by a relevant column to
return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the
few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table.

Only use the following tables:
{table_info}
"""

# Build prompt template from user and system messages
query_prompt_template = ChatPromptTemplate(
    [("system", system_message), ("user", user_prompt)]
)

# Connect to SQLite database
db = SQLDatabase.from_uri('sqlite:///data/freelancer_earnings_bd.db')

# Initialize the model
model = ChatOpenAI(
    model_name=os.getenv('MODEL_NAME'),
    openai_api_base=os.getenv('MODEL_BASE_URL'),
    openai_api_key=os.getenv('MODEL_API_KEY'),
)


def write_query(state: State) -> dict:
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    structured_llm = model.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}


def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}


def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = model.invoke(prompt)
    return {"answer": response.content}


def check_sql(state: State):
    prompt = ("Check if the sql query matches the task.\n\n"
              f'Question: {state["question"]}\n'
              f'SQL Query: {state["query"]}\n'
              )
    structured_llm = model.with_structured_output(Mark)
    response = structured_llm.invoke(prompt)
    return {"check": response["result"]}


# Router function: Decide next step based on whether query is valid
def route(state: State) -> Literal["write_query", "execute_query"]:
    if state["check"]:
        return "execute_query"
    return "write_query"


# Build the graph workflow
graph_builder = StateGraph(State)
graph_builder.add_node(write_query)
graph_builder.add_node(check_sql)
graph_builder.add_node(execute_query)
graph_builder.add_node(generate_answer)
# Define how the nodes are connected
graph_builder.add_edge(START, "write_query")
graph_builder.add_edge("write_query", "check_sql")
graph_builder.add_conditional_edges("check_sql", route)
graph_builder.add_edge("execute_query", "generate_answer")
graph = graph_builder.compile()


# Run the graph for a given question and return the final answer
def run_graph(question: str):
    answer = ''
    for step in graph.stream(
            {"question": question},
            stream_mode="updates"
    ):
        if os.getenv("DEBUG", "false").lower() in ("1", "true", "yes"):
            click.secho(step, fg='yellow')
        answer = step
    return answer['generate_answer']['answer']
