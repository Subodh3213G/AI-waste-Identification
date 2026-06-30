"""Gemini AI Eco-Advisor module for EcoSort AI.

Queries the Google Gemini API to generate structured disposal
instructions and environmental insights.  Falls back to pre-defined
guides when the API is unavailable.
"""
import os
import json
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic schema for the disposal report
# ---------------------------------------------------------------------------

class DisposalGuide(BaseModel):
    """Structured eco-disposal advisory returned by the Gemini API."""
    waste_category: str = Field(description="One of: Organic, Recyclable, Hazardous, E-waste")
    bin_color: str = Field(description="Recommended trash bin color")
    recycling_rules: List[str] = Field(description="3-5 recycling/disposal instructions")
    environmental_impact: str = Field(description="Environmental impact statement")
    decomposition_time: str = Field(description="Estimated decomposition time")
    carbon_tip: str = Field(description="Carbon footprint reduction tip")
    fun_fact: str = Field(description="An interesting environmental fact")


# ---------------------------------------------------------------------------
# Gemini API integration
# ---------------------------------------------------------------------------

def get_disposal_advisory(label: str, waste_category: str) -> Optional[DisposalGuide]:
    """Query Gemini to generate a disposal guide.  Returns None on failure."""
    try:
        import google.generativeai as genai
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key or api_key == "your_gemini_api_key_here":
            return None

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = (
            f"You are an expert environmental waste sorting specialist.\n"
            f"The classified waste object is: {label}\n"
            f"The predicted waste category is: {waste_category}\n\n"
            f"Generate a JSON disposal guide with keys: waste_category, "
            f"bin_color, recycling_rules (list of 3-5 strings), "
            f"environmental_impact, decomposition_time, carbon_tip, fun_fact."
        )

        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=DisposalGuide,
            ),
        )
        data = json.loads(response.text)
        return DisposalGuide(**data)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Fallback guides (offline / no API key)
# ---------------------------------------------------------------------------

_FALLBACK_GUIDES = {
    "Organic": DisposalGuide(
        waste_category="Organic",
        bin_color="Green",
        recycling_rules=[
            "Place in the green compost bin",
            "Remove any stickers, rubber bands, or non-organic attachments",
            "Cut large items into smaller pieces to speed up decomposition",
            "Do not mix with plastic or metal waste",
            "Consider home composting for nutrient-rich soil",
        ],
        environmental_impact=(
            "Organic waste in landfills produces methane, a greenhouse gas "
            "25× more potent than CO₂."
        ),
        decomposition_time="2 weeks – 6 months depending on the item",
        carbon_tip=(
            "Composting at home can reduce your carbon footprint by up to "
            "0.2 tonnes of CO₂ per year."
        ),
        fun_fact=(
            "About 30-40% of the food supply in the US is wasted — "
            "roughly 133 billion pounds each year."
        ),
    ),
    "Recyclable": DisposalGuide(
        waste_category="Recyclable",
        bin_color="Blue",
        recycling_rules=[
            "Rinse the item to remove food residue before recycling",
            "Remove caps and labels where possible",
            "Flatten cardboard boxes to save space",
            "Check local recycling guidelines for accepted materials",
            "Do not bag recyclables in plastic bags",
        ],
        environmental_impact=(
            "Recycling one plastic bottle saves enough energy to power a "
            "60 W light bulb for 6 hours."
        ),
        decomposition_time="Plastic: 450+ years · Glass: 1 million years · Aluminum: 200 years",
        carbon_tip=(
            "Recycling aluminum cans saves 95% of the energy needed to make "
            "new ones from raw materials."
        ),
        fun_fact=(
            "A single recycled plastic bottle can be turned into enough "
            "fiberfill for a ski jacket."
        ),
    ),
    "Hazardous": DisposalGuide(
        waste_category="Hazardous",
        bin_color="Red",
        recycling_rules=[
            "Never place hazardous waste in regular trash bins",
            "Take to a designated hazardous-waste collection facility",
            "Keep in original containers with labels intact",
            "Do not mix different hazardous chemicals together",
            "Contact local waste management for pickup schedules",
        ],
        environmental_impact=(
            "Improper disposal of hazardous waste can contaminate soil and "
            "groundwater for decades."
        ),
        decomposition_time="Varies — many chemicals persist indefinitely in the environment",
        carbon_tip=(
            "Switching to non-toxic, biodegradable cleaning products reduces "
            "hazardous waste at the source."
        ),
        fun_fact=(
            "A single quart of motor oil can contaminate up to "
            "250,000 gallons of drinking water."
        ),
    ),
    "E-waste": DisposalGuide(
        waste_category="E-waste",
        bin_color="Orange / Yellow",
        recycling_rules=[
            "Take to a certified e-waste recycling centre",
            "Remove batteries before recycling and dispose of them separately",
            "Wipe personal data from devices before disposal",
            "Check if the manufacturer offers a take-back programme",
            "Never throw electronics in regular trash — heavy metals leach into soil",
        ],
        environmental_impact=(
            "E-waste represents only 2% of trash in landfills but accounts for "
            "70% of overall toxic waste."
        ),
        decomposition_time="Electronics never fully decompose — metals persist for millions of years",
        carbon_tip=(
            "Extending the life of your phone by just one year saves "
            "~63 kg of CO₂ emissions."
        ),
        fun_fact=(
            "One metric tonne of circuit boards contains 40-800 times more "
            "gold than one tonne of gold ore."
        ),
    ),
}


def get_fallback_guide(waste_category: str) -> DisposalGuide:
    """Return a pre-built guide when the Gemini API is unavailable."""
    return _FALLBACK_GUIDES.get(waste_category, _FALLBACK_GUIDES["Recyclable"])
