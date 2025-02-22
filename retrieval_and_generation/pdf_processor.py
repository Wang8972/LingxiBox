import os
import PyPDF2
from tqdm import tqdm  # 导入tqdm
from sentence_transformers import SentenceTransformer
import torch

class PDFProcessor:
    def __init__(self, data_dir="data", model_name="all-MiniLM-L6-v2"):
        self.data_dir = data_dir
        self.model = SentenceTransformer(model_name)

    def extract_text(self, pdf_path):
        """ 从PDF文件提取文本 """
        text = ""
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text

    def segment_articles(self, text):
        """ 根据标题和语义分割文章 """
        paragraphs = text.split("\n")
        articles = []
        current_article = []
        last_embedding = None

        for paragraph in paragraphs:
            paragraph = paragraph.strip()

            # 跳过空行
            if not paragraph:
                continue

            # 检测标题（通常是短句、独立一行）
            if len(paragraph.split()) <= 10:
                if current_article:
                    articles.append("\n".join(current_article))
                    current_article = []

            # 语义相似度检测，判断是否进入新文章
            embedding = self.model.encode(paragraph, convert_to_tensor=True)
            if last_embedding is not None:
                similarity = torch.nn.functional.cosine_similarity(last_embedding, embedding, dim=0).item()
                if similarity < 0.5 and current_article:
                    articles.append("\n".join(current_article))
                    current_article = []

            current_article.append(paragraph)
            last_embedding = embedding

        if current_article:
            articles.append("\n".join(current_article))

        return articles

    def process_pdfs(self):
        """ 处理所有PDF，返回文章列表 """
        all_articles = []
        for journal in ["Economist", "Atlantic"]:
            folder = os.path.join(self.data_dir, journal)
            if not os.path.exists(folder):
                continue

            pdf_files = [f for f in os.listdir(folder) if f.endswith(".pdf")]
            # 使用 tqdm 显示进度条
            for pdf_file in tqdm(pdf_files, desc=f"Processing {journal}", unit="file"):
                pdf_path = os.path.join(folder, pdf_file)
                text = self.extract_text(pdf_path)
                articles = self.segment_articles(text)
                all_articles.extend([(article, pdf_file) for article in articles])

        return all_articles
