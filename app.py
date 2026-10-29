from langchain_core.runnables import RunnableLambda


# Format the input for the agents
def format_for_agent(x):
    user_input = x["input"] if isinstance(x, dict) else x.input

    return {
        "messages": [("user", user_input)]
    }


# Run both agents
def run_both_agents(x):
    user_input = x["input"] if isinstance(x, dict) else x.input

    # Agent 1: Evaluate the food donation
    evaluation_result = evaluation_chain.invoke({
        "input": user_input
    })

    # Agent 2: Match suitable NGOs and volunteers
    matching_input = (
        "Analyze the following food donation and suggest suitable recipients. "
        "Use the matching tool when sufficient information is available. "
        "Do not invent missing details. If the pickup location or required "
        "donation details are missing, ask the user for them.\n\n"
        f"Donation details:\n{user_input}\n\n"
        f"Food evaluation result:\n{evaluation_result}"
    )

    matching_result = matching_chain.invoke({
        "input": matching_input
    })

    # Return both agent responses
    return (
        "=== AGENT 1: FOOD EVALUATION ===\n"
        f"{evaluation_result}\n\n"
        "=== AGENT 2: NGO / VOLUNTEER MATCHING ===\n"
        f"{matching_result}"
    )


# Create the combined chain
combined_chain = (
    RunnableLambda(format_for_agent)
    | RunnableLambda(
        lambda state: {
            "input": state["messages"][0][1]
        }
    )
    | RunnableLambda(run_both_agents)
).with_types(
    input_type=AgentInput,
    output_type=str
)
