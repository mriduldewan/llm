import requests
from bs4 import BeautifulSoup

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/117.0.0.0 Safari/537.36"
)


class Website:
    """Fetches and parses a webpage, exposing its title and cleaned body text."""

    def __init__(self, url: str, user_agent: str = _DEFAULT_USER_AGENT, timeout: int = 30):
        self.url = url
        response = requests.get(
            url,
            headers={"User-Agent": user_agent},
            timeout=timeout,
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        self.title = soup.title.string if soup.title else "No title found"

        if soup.body:
            for tag in soup.body(["script", "style", "img", "input"]):
                tag.decompose()
            self.text = soup.body.get_text(separator="\n", strip=True)
        else:
            self.text = ""

    def __repr__(self) -> str:
        return f"Website(url={self.url!r}, title={self.title!r})"
