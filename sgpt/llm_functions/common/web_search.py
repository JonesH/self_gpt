import os
import json
import requests
from instructor import OpenAISchema
from pydantic import Field


class Function(OpenAISchema):
    """
    Performs a web search using the Tavily API and returns the top results.
    """

    query: str = Field(..., description="Search query for Tavily.")
    max_results: int = Field(15, description="Maximum number of results to return.")

    class Config:
        title = "web_search"

    @classmethod
    def execute(
        cls,
        query: str,
        max_results: int = 15,
        api_key: str = os.getenv("TAVILY_API_KEY"),
    ) -> str:
        """
        Execute a web search query using the Tavily API.

        Args:
            query (str): The search query.
            max_results (int): The maximum number of results to return.
            api_key (str): The API key for Tavily.

        Returns:
            str: The search results in a formatted string or an error message.
        """
        if not api_key:
            return "Error: Missing API key. Please set the API key as an environment variable."

        try:
            response = cls._send_request(query, max_results, api_key)
            return cls._format_results(response)
        except requests.exceptions.RequestException as e:
            return f"HTTP Request failed: {e}"
        except Exception as e:
            return f"An error occurred: {e}"

    @staticmethod
    def _send_request(query: str, max_results: int, api_key: str) -> dict:
        """
        Send a search request to the Tavily API.

        Args:
            query (str): The search query.
            max_results (int): The maximum number of results to return.
            api_key (str): The API key for Tavily.

        Returns:
            dict: The response from the Tavily API.
        """
        base_url = "https://api.tavily.com/search"
        headers = {"Content-Type": "application/json"}
        payload = {
            "query": query,
            "max_results": max_results,
            "api_key": api_key,
            "search_depth": "basic",
            "topic": "general",
            "days": 7,
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }

        response = requests.post(
            base_url, headers=headers, data=json.dumps(payload), timeout=100
        )
        response.raise_for_status()  # Raises HTTPError for non-2xx responses
        return response.json()

    @staticmethod
    def _format_results(response: dict) -> str:
        """
        Format the search results into a readable string.

        Args:
            response (dict): The response from the Tavily API.

        Returns:
            str: A formatted string of search results or a message if no results are found.
        """
        results = response.get("results", [])
        if not results:
            return "No results found."
        return "\n".join([f"{result['title']}: {result['url']}" for result in results])
