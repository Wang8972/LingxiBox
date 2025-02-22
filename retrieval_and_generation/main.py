import os
from pdf_processor import PDFProcessor
from embedding_indexer import EmbeddingIndexer
from query_handler import QueryHandler


def main():
    processor = PDFProcessor()
    indexer = EmbeddingIndexer()

    if not indexer.load_index():
        print("索引不存在，重新创建...")
        articles = processor.process_pdfs()
        query_handler = QueryHandler(indexer)
        # 创建索引并显示进度条
        indexer.create_index(articles, processor.model)
    else:
        print("索引加载成功！")

    while True:
        query = input("请输入要查询的单词（输入'退出'结束）：")
        if query.lower() == "退出":
            break
        results = query_handler.search(query)
        if results:
            print(f"找到文章来源于：{results[0][1]}")
            print(f"文章内容：\n{results[0][0]}")
        else:
            print("未找到相关文章。")


if __name__ == "__main__":
    main()
