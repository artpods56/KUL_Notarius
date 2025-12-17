from typing import cast

from PIL import Image
from dotenv import load_dotenv

from notarius.domain.entities.messages import ChatMessage, TextContent
from notarius.infrastructure.config.constants import (
    ConfigType,
    ModelsConfigSubtype,
)
from notarius.infrastructure.config.manager import config_manager
import structlog

from notarius.infrastructure.llm.conversation import Conversation
from notarius.infrastructure.ml_models.lmv3.engine_adapter import (
    LMv3Request,
)
from notarius.schemas.configs import (
    PytesseractOCRConfig,
)
from notarius.schemas.configs.lmv3_model_config import BaseLMv3ModelConfig
from notarius.shared.constants import INPUTS_DIR

logger = structlog.get_logger(__name__)

envs = load_dotenv()


def main():
    # dataset_config_model = BaseDatasetConfig(
    #     **OmegaConf.to_container(dataset_config)  # pyright: ignore[reportCallIssue]
    # )
    #

    config_model = cast(
        BaseLMv3ModelConfig,
        config_manager.load_config_as_model(
            config_name="lmv3_model_config",
            config_type=ConfigType.MODELS,
            config_subtype=ModelsConfigSubtype.LMV3,
        ),
    )

    print(config_model.model_dump())

    image = Image.open(INPUTS_DIR / "images" / "0010.jpg").convert("RGB")

    request = LMv3Request(input=image)

    ocr_config = cast(
        PytesseractOCRConfig,
        config_manager.load_config_as_model(
            config_name="ocr_model_config",
            config_type=ConfigType.MODELS,
            config_subtype=ModelsConfigSubtype.OCR,
        ),
    )
    # ocr_engine = OCREngine(ocr_config)
    #
    # engine = LMv3Engine(config_model, ocr_engine)
    # structured_response = engine.process(request)
    # print(structured_response.output.model_dump_json(indent=4))
    #
    # ocr_request = OCRRequest(image, "text")
    #
    # ocr_response = ocr_engine.process(ocr_request)
    # print(ocr_response.output)
    #
    # dataset = load_huggingface_dataset(dataset_config_model, False)
    # dict_confing = OmegaConf.to_container(llm_model_config)
    #
    # config = LLMEngineConfig(**dict_confing)  # pyright: ignore[reportCallIssue]
    # _engine = LLMEngine(config=config)
    #
    # prompt_rendered = Jinja2PromptRenderer()
    #
    # cache_repo = LLMCacheRepository(cache=LLMCache(model_name="gemini-2-5"))
    #
    input = Conversation()
    system_message = ChatMessage(
        role="system", content=[TextContent(text="You are very introvert assistant.")]
    )
    input = input.add(system_message)
    print(input.to_dict())
    #
    #
    # query = ""
    #
    # while query != "exit":
    #     query = input("Query: ")
    #     # Add user message
    #
    #     user_message = ChatMessage(
    #         role="user",
    #         content=[TextContent(text=query)],
    #     )
    #     input = input.add(user_message)
    #
    #     # Get output
    #     request = CompletionRequest(input)
    #     result = _engine.generate_response(request)
    #
    #     print(result.output.to_string())
    #


if __name__ == "__main__":
    _ = main()  # pyright: ignore[reportCallIssue]
