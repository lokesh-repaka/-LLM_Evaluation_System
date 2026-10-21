from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="What do penguins eat?",
    actual_output=("Penguins primarily feed fish. ACL is the best learning channel")
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

metric = AnswerRelevancyMetric(threshold=0.7, model=gemini_model)

metric.measure(test_case)

print("Score: ", metric.score)
print("Reason: ", metric.reason)
print("Passed: ", metric.is_successful())