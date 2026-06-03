from tools.base_web_tool_monaco import BaseWebToolMonaco


class VisitWebpage(BaseWebToolMonaco):
    """
    This is a tool to simulate access to a webpages.
    """

    name = "visit_webpage"
    description = (
        "Navigates to a specific URL and extracts the textual content of the webpage. "
    )
    output_type = "string"
    inputs = {
        "url": {
            "type": "string",
            "description": "The full URL of the webpage to visit and retrieve information from."
        }
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.web_search_tool = None

        self._generation_prompts = {}

    def execute_tool(self, url: str) -> str:
        """
        Returns the content of the Wikipedia page at the given URL.

        Arguments:
            url (str) -- The URL to the Wikipedia page requested by the agent. This must be a URL taken from the
             Wikipedia dump.
        """
        try:
            # Retrieve the Wikipedia page at the current URL
            page_content = self._monaco_utils.retrieve_wiki_page(
                wiki_url=url,
                monaco_entry_to_remove=None,
                integrate_infobox=True, integrate_lists_tables=True).wiki_page

        except ValueError:
            raise ValueError(f'Unable to retrieve webpage at URL "{url}".')

        return page_content
