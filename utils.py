
from langchain_core.output_parsers import StrOutputParser
import re
class Parser(StrOutputParser):
    def parse(self, output: str) -> str:
        """
        This method will parse the string output and extract the query from the 'Query' section.

        Args:
            output (str): The full output string containing Context, Question, Reasoning, and Query.

        Returns:
            str: The parsed query extracted from the output.
        """
        question_matches = re.findall(r'(Query|Decision|Answer)[ :]+["\']?(.+)["\']?', output, re.IGNORECASE | re.MULTILINE)
        if question_matches:
            return question_matches[-1][-1]
        else:
            raise ValueError("Query|Decision|Answer section not found in output.")

def deduplicate(seq: list[str]) -> list[str]:

    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)