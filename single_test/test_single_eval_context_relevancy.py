from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase


test_case = LLMTestCase(
    input="Where do penguins live?",
    actual_output="Penguins live in a variety of environments in the Southern Hemisphere. They are commonly associated with the icy coasts of Antarctica, but they also inhabit the cool shores of South America, southern Africa, and even the Galapagos Islands near the equator.",
    retrieval_context=["Penguins are flightless seabirds that are specially adapted for life in the Southern Hemisphere."]
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

metric = ContextualRelevancyMetric(threshold=0.7, model=gemini_model)

metric.measure(test_case)


print("Score: ", metric.score)
print("Reason: ", metric.reason)
print("Passed: ", metric.is_successful())