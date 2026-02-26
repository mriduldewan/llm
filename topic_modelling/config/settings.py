import os
from dotenv import load_dotenv

load_dotenv(override=True)

# --- OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano-2025-04-14")

# --- Ollama ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL_DEEPSEEK = os.getenv("OLLAMA_MODEL_DEEPSEEK", "deepseek-r1:8b")
OLLAMA_MODEL_GPT_OSS = os.getenv("OLLAMA_MODEL_GPT_OSS", "gpt-oss:20b")

# --- Concurrency ---
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "4"))

# --- Paths ---
INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "input", "topic_model_batch_inputs")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "topic_model_batch_outputs")

# --- Topic list ---
TOPICS = [
    "Enrolment Process",
    "Student Support Services",
    "Course Content and Relevance",
    "Trainer Quality and Engagement",
    "Facilities and Campus Environment",
    "Timetable and Scheduling",
    "Online Learning Platform",
    "Assessment and Feedback",
    "Career and Employment Services",
    "Technology and Equipment",
    "Communication and Information",
    "Student Welfare and Wellbeing",
    "Course Fees and Payments",
    "Recognition of Prior Learning (RPL)",
    "Work Placement",
    "Graduation and Completion",
]
