from openai.types.responses import Response
from pydantic import BaseModel


def get_text_content(response: BaseModel | Response) -> str:
    if isinstance(response, Response):
        return response.output_text
    return response.model_dump_json()
