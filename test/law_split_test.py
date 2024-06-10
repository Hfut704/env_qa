from docx import Document
import re



def read_docx(file_path):
    # 加载文档
    doc = Document(file_path)
    full_text = []
    # 遍历文档中的每个段落，并将其文本内容添加到列表中
    for para in doc.paragraphs:
        full_text.append(para.text)
    # 将列表中的文本内容连接成一个字符串，并返回
    return '\n'.join(full_text)


def get_intro(lines_without_title):
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

def get_chapters(lines):
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

def get_catalog(lines):
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

def get_article(lines):
    articles = []
    content = ""
    for line in lines:
        if re.match(r"^第.*条", line):
            articles.append(content)
            content=''
        content += line+'\n'
    articles = [l for l in articles if l != '']
    return  articles


def preprocess(doc_text):
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

def get_title(lines):
    return lines[0], lines[1:]

def split(file_path):
    # 读取文件并打印内容
    doc = {}
    doc_text = read_docx(file_path)
    lines = preprocess(doc_text)
    title, lines = get_title(lines)
    intro, content_lines = get_intro(lines)
    catalog, content_lines = get_catalog(content_lines)
    doc['title'] = title
    doc['intro'] = intro
    doc['catalog'] = catalog
    chapters = get_chapters(content_lines)
    if len(chapters) == 0:
        articles = get_article(content_lines)
        doc['articles'] = articles
    else:
        for chapter in chapters:
            lines = chapter['content'].split('\n')
            articles = get_article(lines)
            chapter['articles'] = articles
        doc['chapters'] = chapters
    return doc
d = split('E:\desktop\\章-条.docx')
d





