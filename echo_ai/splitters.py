import re
from typing import Sequence, Any, List

from langchain.document_loaders import PyPDFLoader

from langchain.schema import Document
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import TextSplitter
from langchain.chat_models import ChatOpenAI
from .output_parsers import MyParser
import re
def split_with_delimiters(text: str, delimiters: List[str]) -> List[str]:
    """
    按照给出的字符列表对text进行切分，保留切分字符串作为前一个句子的结尾。
    :param text:
    :param delimiters:
    :return:
    """
    pattern = '|'.join(re.escape(s) for s in delimiters)
    # 使用正则表达式进行切分，并保留切分的字符串
    result = re.split(f'({pattern})', text)
    # 合并相邻的切分字符和切分结果
    result = [''.join(result[i:i + 2]) for i in range(0, len(result), 2)]
    return result


class ChineseTextSplitter(TextSplitter):
    """
    中文分割器
    """
    def __init__(self, separators: List[str], num_sentences=1, overlap_sentences=0, **kwargs: Any):
        """Create a new TextSplitter."""
        super().__init__(**kwargs)

        self.separators = separators
        self.overlap_sentences = overlap_sentences
        self.num_sentences = num_sentences

    def split_text(self, text: str) -> List[str]:
        """"
        按照分割符号进行切分，每个片段保留前后overlap_sentences个句子
        """

        segments = split_with_delimiters(text, self.separators)
        left_point = 0
        right_point = 2 * self.overlap_sentences + self.num_sentences

        docs = []

        # 使用一个滑动窗口对句子进行合并，保留重叠部分
        while right_point < len(segments):
            temp_text = ""
            for i in range(left_point, right_point):
                temp_text += segments[i]
            docs.append(temp_text)
            left_point += self.overlap_sentences + + self.num_sentences
            right_point += self.overlap_sentences + + self.num_sentences

        temp_text = ""
        while left_point < len(segments):
            temp_text += segments[left_point]
            left_point += 1
        docs.append(temp_text)
        return docs

    async def atransform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        pass


class SemanticsTextSplitter(TextSplitter):
    """
    文本语义分割器
    """
    def __init__(self, mode='less', **kwargs: Any):
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        self.llm = ChatOpenAI(temperature=0.1, model_name='gpt-3.5-turbo')
        self.parser = MyParser()

    def split_text(self, text: str) -> List[str]:
        """
        使用大模型对文本按照语义进行切割
        :param text:
        :return:
        """

        template = f"""
               你是一个信息抽取助手，你需要将我给出的文本段按照语义进行切割，不能返回除了给出文本以外的内容。
               以下是一个示例：
               待切割文本段：
               '''
               如何使用langchain？
               1) 导入相关的包。
               2) 在项目中使用相关的代码。
               3) 返回结果。
               如何使用langchain.llm进行问答？
               见官方文档。
               '''
               切割结果：
               ###
                如何使用langchain？
               1) 导入相关的包。
               2) 在项目中使用相关的代码。
               3) 返回结果。
               ###
               ###
               如何使用langchain.llm进行问答？
               见官方文档。
               ###
               """
        system_message = SystemMessage(
            content=template)
        human_message = HumanMessage(content=f"""待切割文本段：{text}""")
        answer = self.llm.predict_messages([system_message, human_message])
        res = self.parser.parse(text=answer.content)
        return res

    async def atransform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        pass


class ChineseLawSplitter(TextSplitter):
    def __init__(self, **kwargs: Any):
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
    def split_text(self, text: str) -> List[str]:
        """
        对法律法规按照标题，介绍，目录，章，条进行切割
        :param text:
        :return:
        """
        self.split(text)

    def read_docx(self, file_path):
        from docx import Document as Documentx
        # 加载文档
        doc = Documentx(file_path)
        full_text = []
        # 遍历文档中的每个段落，并将其文本内容添加到列表中
        for para in doc.paragraphs:
            full_text.append(para.text)
        # 将列表中的文本内容连接成一个字符串，并返回
        return '\n'.join(full_text)


    def get_intro(self, lines_without_title):
        intro = ""
        start = 0
        for line in lines_without_title:
            if re.match(r"^目录", line):
                break
            if re.match(r"^第.*章", line):
                break
            if re.match(r"^第.*条", line):
                break
            intro+=line+'\n'
            start += 1
        return intro, lines_without_title[start:]

    def get_chapters(self, lines):
        chapters = []
        chapter = {}
        content = ""
        title = ""
        for line in lines:
            if re.match(r"^第.*?章", line):
                chapter = {}
                chapter['title'] = title
                chapter['content'] = content
                chapters.append(chapter)
                title = line
                content = ""
            else:
                content += line+'\n'

        chapter = {}
        chapter['title'] = title
        chapter['content'] = content
        chapters.append(chapter)
        del chapters[0]
        return chapters

    def get_catalog(self, lines):
        catalog = ""
        start = 0
        n = 0
        for line in lines:
            if re.match(r"^第一章", line):
                if n > 0:
                    break
                n += 1
            if re.match(r"^第.*条", line):
                break
            catalog += line+'\n'
            start += 1
        if n < 2:
            return "", lines
        return catalog, lines[start:]

    def get_article(self, lines):
        articles = []
        content = ""
        for line in lines:
            if re.match(r"^第.*条", line):
                articles.append(content)
                content=''
            content += line+'\n'
        articles = [l for l in articles if l != '']
        return  articles


    def preprocess(self, doc_text):
        lines = doc_text.split('\n')
        temp = []
        for line in lines:
            line = re.sub(r'\s+', '', line)
            line = re.sub(r'(第.*?章)', r'\1 ', line)
            line = re.sub(r'(第.*?条)', r'\1 ', line)
            if line != "":
                temp.append(line)
        lines = temp
        return lines

    def get_title(self, lines):
        return lines[0], lines[1:]

    def split(self, file_path):
        # 读取文件并打印内容
        doc = {}
        doc_text = self.read_docx(file_path)
        lines = self.preprocess(doc_text)
        title, lines = self.get_title(lines)
        intro, content_lines = self.get_intro(lines)
        catalog, content_lines = self.get_catalog(content_lines)
        doc['title'] = title
        doc['intro'] = intro
        doc['catalog'] = catalog
        chapters = self.get_chapters(content_lines)
        if len(chapters) == 0:
            articles = self.get_article(content_lines)
            doc['articles'] = articles
        else:
            for chapter in chapters:
                lines = chapter['content'].split('\n')
                articles = self.get_article(lines)
                chapter['articles'] = articles
            doc['chapters'] = chapters
        return doc


if __name__ == '__main__':
    import os
    os.environ["OPENAI_API_KEY"] = ""
    documents = PyPDFLoader('../../../intelligent-qa/data/test.pdf').load()
    splitter = SemanticsTextSplitter()
    res = splitter.split_documents(documents)
    res
