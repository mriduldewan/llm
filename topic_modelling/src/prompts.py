from config.settings import TOPICS

_TOPIC_LIST = "\n".join(f"- {t}" for t in TOPICS)

SYSTEM_PROMPT = f"""You are an expert topic classifier for student feedback from a vocational education and training institution.
Your task is to analyze student verbatims and assign one or more topics from a predefined list.You must adhere to the following rules:
1.  Match the verbatim to the most relevant topics from the provided topic list.
2.  If the verbatim is not relevant to any topic on the list, return 'No Match'.
3.  You can assign more than one topic if the verbatim covers multiple subjects.
4.  Your output must be a single JSON object.
5.  The JSON object must have two keys: 'topics' and 'verbatim_text'.
6.  The value for 'topics' should be a list of strings. Each string must be a topic from the provided list or the string 'No Match'.
7.  The value for 'verbatim_text' should be the original verbatim you are analyzing.

Below is a list of predefined topics and five examples of how to classify a verbatim.

**Topic List:**
{_TOPIC_LIST}"""


def user_prompt_for(verbatim: str) -> str:
    return f"""
    You are looking at a verbatim from a student. Based on the list of topics provided, below are five examples of how to classify a verbatim.

    **Examples for Few-Shot Classification:**

    1. **Verbatim:** "I'm having trouble with the login for the online portal, and the Wi-Fi on campus is really slow."
       **Topics:** ["Online Learning Platform", "Technology and Equipment"]

    2. **Verbatim:** "John is great! He explains everything clearly and is always available to help after class."
       **Topics:** ["Trainer Quality and Engagement"]

    3. **Verbatim:** "I asked about my results from last semester, but nobody has gotten back to me. I've been waiting for weeks."
       **Topics:** ["Communication and Information", "Assessment and Feedback"]

    4. **Verbatim:** "The campus cafeteria has really limited options, and the library hours are not great for students who work."
       **Topics:** ["Facilities and Campus Environment"]

    5. **Verbatim:** "I've been working as a mechanic for 10 years, and I want to see if I can get credit for my experience towards this course."
       **Topics:** ["Recognition of Prior Learning (RPL)"]

    **New Verbatim to Classify:**
    {verbatim}
    """


def messages_for(verbatim: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt_for(verbatim)},
    ]
