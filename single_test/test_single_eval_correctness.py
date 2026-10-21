from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams
import json

test_case = LLMTestCase(
    input="Where do penguins live?",
    actual_output="Penguins live in a variety of environments in the Southern Hemisphere. They are commonly associated with the icy coasts of Antarctica, but they also inhabit the cool shores of South America, southern Africa, and even the Galapagos Islands near the equator.",
    expected_output="Penguins live in the Southern Hemisphere, including Antarctica, South America, southern Africa, and the Galapagos Islands."
)

from deepeval.models import GeminiModel
from dotenv import load_dotenv
import os

load_dotenv()

# Use Gemini as the judge model for evaluation
gemini_model = GeminiModel(
    model="gemini-3.5-flash-lite",
    api_key=os.getenv("GEMINI_API_KEY")
)

metric = GEval(
    name="Correctness",
    criteria=(
        "Determine whether the actual output is factually correct based on the expect output"
    ),
    evaluation_params=[
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT
    ],
    model=gemini_model
)

metric.measure(test_case=test_case)

print("Score: ", metric.score)
print("Reason: ", metric.reason)