# from langchain_openai import AzureChatOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_community.chat_models import AzureChatOpenAI
from deepeval.models import AzureOpenAIModel


class AzureOpenAI(DeepEvalBaseLLM):
    def __init__(
        self,
        model
    ):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Custom Azure OpenAI Model"
