
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

app = FastAPI(title="Food Rescue AI Agent")

# Demo recipients. These are not verified real organizations.
RECIPIENTS = [
    {
        "name": "Hope Community Kitchen (Demo)",
        "location": "Hyderabad",
        "capacity": 120,
        "categories": ["rice", "cooked meals", "vegetarian", "vegetables"],
    },
    {
        "name": "Helping Hands Volunteer Group (Demo)",
        "location": "Secunderabad",
        "capacity": 45,
        "categories": ["bread", "snacks", "packaged food", "fruits"],
    },
    {
        "name": "Green Plate Foundation (Demo)",
        "location": "Madhapur, Hyderabad",
        "capacity": 80,
        "categories": ["vegetarian", "fruits", "vegetables", "packaged food"],
    },
]


class Donation(BaseModel):
    food_type: str
    quantity: int = Field(gt=0)
    prepared_hours_ago: float = Field(ge=0)
    storage: str
    hours_until_expiry: float = Field(ge=0)
    pickup_location: str


def get_response(result):
    messages = result.get("messages", [])
    for message in reversed(messages):
        content = getattr(message, "content", "")
        if isinstance(content, str) and content:
            return content
        if isinstance(content, list):
            texts = [
                item["text"]
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            if texts:
                return "\n".join(texts)
    return "No response was returned by the agent."


def build_agents():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not configured.")

    model = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0,
        google_api_key=api_key,
    )

    evaluation_agent = create_agent(
        model=model,
        tools=[],
        system_prompt=(
            "You are a food donation evaluation agent. Assess urgency using "
            "the supplied details. Never claim food is safe merely from its age "
            "or storage description. Flag room-temperature cooked food for "
            "prompt food-safety verification. Recommend checking actual handling, "
            "temperature, packaging, and applicable food-safety guidance. "
            "Give a concise response."
        ),
    )

    matching_agent = create_agent(
        model=model,
        tools=[],
        system_prompt=(
            "You are an NGO matching agent. Match the donation to the supplied "
            "demo recipient records by food category and pickup location. "
            "Only use the records provided. State clearly that all recipients "
            "are demo records, not verified organizations. Never claim that "
            "pickup or acceptance is confirmed."
        ),
    )

    return evaluation_agent, matching_agent


@app.get("/")
def home():
    return {
        "message": "Food Rescue AI Agent is running!",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/rescue")
def rescue_food(donation: Donation):
    try:
        evaluation_agent, matching_agent = build_agents()

        details = donation.model_dump()
        donation_text = "\n".join(
            f"{key.replace('_', ' ').title()}: {value}"
            for key, value in details.items()
        )

        evaluation_result = evaluation_agent.invoke({
            "messages": [
                {"role": "user", "content": donation_text}
            ]
        })
        evaluation = get_response(evaluation_result)

        matching_prompt = f"""
Donation details:
{donation_text}

Food safety and urgency assessment:
{evaluation}

Demo recipient records:
{RECIPIENTS}

Find the most appropriate possible match based on the food category
and pickup location. Explain that the records are only examples.
"""

        matching_result = matching_agent.invoke({
            "messages": [
                {"role": "user", "content": matching_prompt}
            ]
        })
        matching = get_response(matching_result)

        return {
            "donation": details,
            "food_evaluation": evaluation,
            "ngo_matching": matching,
            "notice": (
                "Recipient records are demonstrations only. "
                "Verify food safety and recipient availability before donation."
            ),
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Food rescue workflow failed: {str(error)}",
        )
