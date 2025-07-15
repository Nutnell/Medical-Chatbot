import json
from typing import Optional, List, Any, Annotated
from pydantic import Field
from pydantic.functional_validators import AfterValidator
from langchain_core.language_models.llms import LLM

class MetaBedrockLLM(LLM):
    bedrock_client: Annotated[Any, AfterValidator(lambda x: x)] = Field(...)
    model_id: str = Field(...)
    temperature: float = Field(default=0.8)
    max_gen_len: int = Field(default=128)  # Lowered to reduce looping

    @property
    def _llm_type(self):
        return "custom-meta-bedrock"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        payload = {
            "prompt": prompt,
            "temperature": self.temperature,
            "max_gen_len": self.max_gen_len
        }
        response = self.bedrock_client.invoke_model(
            body=json.dumps(payload),
            modelId=self.model_id,
            accept="application/json",
            contentType="application/json"
        )
        result = json.loads(response['body'].read())
        generation = result.get("generation", "No response")

        # Truncate overly verbose output
        return ". ".join(generation.split(". ")[:5]) + "."
