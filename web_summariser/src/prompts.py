from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .scraper import Website

SYSTEM_PROMPT = (
    "You are an assistant that analyzes the contents of a website "
    "and provides a short summary, ignoring text that might be navigation related. "
    "Don't be too verbose and don't mention your thoughts; stick to the content. "
    "Respond in markdown."
)


def user_prompt_for(website: "Website") -> str:
    prompt = f"You are looking at a website titled: {website.title}\n\n"
    prompt += (
        "The contents of this website are as follows. "
        "Please provide a short summary in markdown. "
        "If the page includes news or announcements, summarise those too.\n\n"
    )
    prompt += website.text
    return prompt


def messages_for(website: "Website") -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt_for(website)},
    ]
