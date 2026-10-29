import os
from typing import Any

import uvicorn
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field


# ---------------------------------------------------------
# 1. Tools used by the two agents
# ---------------------------------------------------------

@tool
def evaluate_food_donation(
    food_name: str,
    food_type: str,
    quantity: int,
    use_by_hours: float,
    prepared_hours_ago: float,
    storage_notes: str = ""
) -> str:
    """Evaluate food donation urgency and flag details needing a human safety review.

    Provide the food name/type, quantity, hours remaining until the use-by time,
    hours since preparation, and any storage notes.
    This tool does not certify food as safe.
    """
    if quantity <= 0:
        return "REJECT FOR REVIEW: Quantity must be greater than zero."

    if use_by_hours <= 0:
        urgency = "Critical"
        action = "Do not distribute food past its use-by time. Arrange safe disposal."
    elif use_by_hours <= 3:
        urgency = "High"
        action = "Arrange prompt pickup and confirm safe handling."
    elif use_by_hours <= 8:
        urgency = "Medium"
        action = "Find a recipient soon and confirm collection timing."
    else:
        urgency = "Low"
        action = "There is more time to coordinate collection."

    flags = []
    if food_type.lower() in ("cooked meals", "cooked food") and prepared_hours_ago > 2:
        flags.append(
            "Cooked food has been held for over 2 hours; verify time, temperature, "
            "and local food-safety guidance before distribution."
        )

    notes_lower = storage_notes.lower()
    if any(word in notes_lower for word in ("spoiled", "bad smell", "unsafe", "unknown temperature")):
        flags.append("Donor notes indicate a possible safety concern. Do not distribute until reviewed.")

    if flags:
        safety = " HUMAN REVIEW REQUIRED: " + " ".join(flags)
    else:
        safety = (
            " No obvious warning was identified from the supplied fields, but this is not "
            "a food-safety certification. A person must verify storage and handling."
        )

    return (
        f"Food: {food_name}; type: {food_type}; quantity: {quantity}. "
        f"Urgency: {urgency}. Recommendation: {action}{safety}"
    )


@tool
def match_ngos_and_volunteers(
    food_type: str,
    food_name: str,
    quantity: int,
    pickup_location: str,
    use_by_hours: float
) -> str:
    """Suggest demo NGO/volunteer matches for a food donation.

    Match using food needs, location, capacity, and urgency. The sample organizations
    are demonstration records only and are not verified real-world contacts.
    """
    recipients = [
        {
            "name": "Hope Community Kitchen (demo)",
            "type": "Community kitchen",
            "location": "Hyderabad",
            "capacity": 120,
            "needs": "cooked meals, vegetarian, rice, vegetables"
        },
        {
            "name": "Helping Hands Volunteer Group (demo)",
            "type": "Volunteer group",
            "location": "Secunderabad",
            "capacity": 45,
            "needs": "bread, snacks, packaged food, fruits"
        },
        {
            "name": "Green Plate Foundation (demo)",
            "type": "NGO",
            "location": "Madhapur, Hyderabad",
            "capacity": 80,
            "needs": "vegetarian, fruits, vegetables, packaged food"
        }
    ]

    food_text = f"{food_type} {food_name}".lower()
    pickup_text = pickup_location.lower()
    matches = []

    for recipient in recipients:
        score = 0
        reasons = []
        recipient_location = recipient["location"].lower()

        if any(part in pickup_text or part in recipient_location for part in ["hyderabad", "madhapur", "secunderabad"]):
            if "hyderabad" in pickup_text and "hyderabad" in recipient_location:
                score += 35
                reasons.append("same city area")
            elif recipient_location in pickup_text or pickup_text in recipient_location:
                score += 40
                reasons.append("location match")
            else:
                score += 15
                reasons.append("nearby area; travel distance must be confirmed")
        else:
            score += 5
            reasons.append("location needs manual confirmation")

        needs = recipient["needs"]
        if any(word in food_text for word in needs.split(", ")):
            score += 30
            reasons.append("food type appears to match recipient needs")
        else:
            score += 10
            reasons.append("confirm that the recipient can accept this food type")

        if quantity <= recipient["capacity"]:
            score += 20
            reasons.append("listed capacity is sufficient")
        else:
            score += 5
            reasons.append("quantity exceeds listed capacity; consider splitting donation")

        if use_by_hours <= 3:
            score += 10
            reasons.append("high urgency; quick collection needed")

        matches.append((score, recipient, reasons))

    matches.sort(key=lambda item: item[0], reverse=True)
    lines = [
        "Suggested matches (sample/demo records only; verify organizations and availability before use):"
    ]
    for score, recipient, reasons in matches:
        lines.append(
            f"- {recipient['name']} | type: {recipient['type']} | area: {recipient['location']} "
            f"| listed capacity: {recipient['capacity']} | score: {min(score, 100)}/100 "
            f"| reasons: {', '.join(reasons)}"
        )
    lines.append(
        "These are heuristic suggestions, not confirmed bookings. Contact and verify a recipient "
        "before arranging collection."
    )
    return "\n".join(lines)


# ---------------------------------------------------------
# 2. Initialize the model and two specialized agents
# ---------------------------------------------------------

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_APIKEY")
if not GOOGLE_API_KEY:
    raise RuntimeError(
        "Please set the GOOGLE_API_KEY environment variable before starting the app."
    )

llm = ChatGoogleGenerativeAI(
    model=os.environ.get("GOOGLE_MODEL", "gemini-3.8-flash"),
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)

food_evaluation_agent = create_agent(
    model=llm,
    tools=[evaluate_food_donation],
    system_prompt=(
        "You are the Food Evaluation Agent for a food rescue service. "
        "Collect the details needed to evaluate a donation and use the evaluate_food_donation tool. "
        "Clearly report urgency, recommendation, and any safety flags. Never claim food is safe "
        "based only on these inputs. If the use-by time has passed, advise against distribution. "
        "Do not invent facts."
    )
)

ngo_matching_agent = create_agent(
    model=llm,
    tools=[match_ngos_and_volunteers],
    system_prompt=(
        "You are the NGO Matching Agent for a food rescue service. "
        "Use the match_ngos_and_volunteers tool to rank possible recipients. "
        "Explain the match reasons and remind users that the listed organizations are demo records "
        "that must be verified. Never claim that a recipient has accepted a donation."
    )
)


# ---------------------------------------------------------
# 3. Request and response formatting for LangServe
# ---------------------------------------------------------

class AgentInput(BaseModel):
    input: str = Field(description="Describe the food donation or ask the food rescue agent a question")


def format_for_agent(x: Any) -> dict:
    user_input = x["input"] if isinstance(x, dict) else x.input
    return {"messages": [("user", user_input)]}



def extract_text_response(agent_output: Any) -> str:
    if not isinstance(agent_output, dict):
        return str(agent_output)

    messages = agent_output.get("messages", [])

    for message in reversed(messages):
        content = getattr(message, "content", None)

        if content is None and isinstance(message, dict):
            content = message.get("content")

        if isinstance(content, str) and content.strip():
            return content

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            if parts:
                return "\n".join(parts)

    return str(agent_output)



def make_agent_chain(agent):
    return (
        RunnableLambda(format_for_agent)
        | agent
        | RunnableLambda(extract_text_response)
    ).with_types(input_type=AgentInput, output_type=str)


evaluation_chain = make_agent_chain(food_evaluation_agent)
matching_chain = make_agent_chain(ngo_matching_agent)


def run_both_agents(x: Any) -> str:
    """Run the evaluation agent first, then the matching agent."""
    user_input = x["input"] if isinstance(x, dict) else x.input

    evaluation_result = evaluation_chain.invoke({"input": user_input})
    matching_result = matching_chain.invoke({
        "input": (
            "Use the donation details from this user message and the evaluation result to suggest "
            "recipients. If details needed by the matching tool are missing, ask for them instead "
            "of guessing.\n\nUser's donation details:\n"
            f"{user_input}\n\nFood Evaluation Agent result:\n{evaluation_result}"
        )
    })

    return (
        "=== AGENT 1: FOOD EVALUATION ===\n"
        f"{evaluation_result}\n\n"
        "=== AGENT 2: NGO / VOLUNTEER MATCHING ===\n"
        f"{matching_result}"
    )


combined_chain = (
    RunnableLambda(format_for_agent)
    | RunnableLambda(lambda state: {"input": state["messages"][0][1]})
    | RunnableLambda(run_both_agents)
).with_types(input_type=AgentInput, output_type=str)


# ---------------------------------------------------------
# 4. FastAPI application and LangServe routes
# ---------------------------------------------------------

app = FastAPI(
    title="Food Rescue AI Agent API",
    description=(
        "Two-agent food rescue demo: evaluate donation urgency and suggest NGO/volunteer matches."
    ),
    version="1.0.0"
)

add_routes(app, evaluation_chain, path="/food-evaluation-agent", playground_type="default")
add_routes(app, matching_chain, path="/ngo-matching-agent", playground_type="default")
add_routes(app, combined_chain, path="/agent", playground_type="default")


@app.get("/")
def home():
    return {
        "message": "Food Rescue AI Agent API is running",
        "agents": {
            "food_evaluation": "/food-evaluation-agent/playground/",
            "ngo_matching": "/ngo-matching-agent/playground/",
            "combined_workflow": "/agent/playground/"
        },
        "docs": "/docs"
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
