from retry import retry
from pyorigin.config.init_class import InitClass
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class BigModel:
    def __init__(self):
        self.model = InitClass().get_model()

    @retry(tries=3)
    def single_invoke(self, content, template, examples="", output_parser=StrOutputParser(), history=""):
        model = self.model
        prompt = PromptTemplate.from_template(
            str(template)
        ).partial(examples=examples, history=history)
        output_text = (prompt | model | output_parser).invoke(input=content)
        return output_text



if __name__ == "__main__":
    template = """
    content: {content}
    examples:
    content: 你好\noutput: 杂鱼
    content: 你好\noutput: 杂鱼
    content: 你好\noutput: 杂鱼
    content: 你好\noutput: 杂鱼
    """
    print(BigModel.single_invoke(content="你是谁", template=template))