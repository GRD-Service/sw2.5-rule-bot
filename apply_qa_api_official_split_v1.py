from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='qa_api.py')
    parser.add_argument('--output', default='qa_api_official_split_v1.py')
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    text = src.read_text(encoding='utf-8')

    old_init = '''all_index_documents = list(db.docstore._dict.values())

# logical_pageを持たない表紙・カバー等はstrict retrievalから除外する。
search_documents = [
    doc
    for doc in all_index_documents
    if doc.metadata.get("logical_page") is not None
]

page_documents_by_pdf = defaultdict(list)
page_documents_by_logical = defaultdict(list)
logical_to_pdf = {}

for doc in search_documents:
'''

    new_init = '''all_index_documents = list(db.docstore._dict.values())

# ------------------------------------------------------------
# Document classes
# ------------------------------------------------------------
#
# 同じFAISSには書籍本文と公式エラッタを格納するが、ページ単位の
# navigation / reference / structured search に公式エラッタを混ぜない。
#
# 旧インデックスとの互換性のため source_class 未設定は book とみなす。
book_documents = [
    doc
    for doc in all_index_documents
    if doc.metadata.get("source_class", "book") == "book"
]

official_documents = [
    doc
    for doc in all_index_documents
    if doc.metadata.get("source_class") == "official_correction"
]

# logical_pageを持たない表紙・カバー等はstrict retrievalから除外する。
# 既存コード中の search_documents は「検索可能な書籍本文」を意味する。
search_documents = [
    doc
    for doc in book_documents
    if doc.metadata.get("logical_page") is not None
]

print(
    "Index documents: "
    f"all={len(all_index_documents)}, "
    f"book={len(book_documents)}, "
    f"book_searchable={len(search_documents)}, "
    f"official={len(official_documents)}"
)

page_documents_by_pdf = defaultdict(list)
page_documents_by_logical = defaultdict(list)
logical_to_pdf = {}

for doc in search_documents:
'''

    if old_init not in text:
        raise RuntimeError('FAISS/document initialization block was not found. qa_api.py may have changed.')
    text = text.replace(old_init, new_init, 1)

    old_lexical = '''lexical_matrix = lexical_vectorizer.fit_transform(
    [doc.page_content for doc in search_documents]
)


# ============================================================
# Book categories / authority
# ============================================================
'''

    new_lexical = '''lexical_matrix = lexical_vectorizer.fit_transform(
    [doc.page_content for doc in search_documents]
)

# 公式エラッタは書籍本文とは別の語彙空間で検索する。
# 現段階では /ask のcontextにはまだ合流させない。
official_lexical_vectorizer = None
official_lexical_matrix = None

if official_documents:
    official_lexical_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        min_df=1,
    )
    official_lexical_matrix = official_lexical_vectorizer.fit_transform(
        [doc.page_content for doc in official_documents]
    )


# ============================================================
# Book categories / authority
# ============================================================
'''

    if old_lexical not in text:
        raise RuntimeError('Lexical index block was not found. qa_api.py may have changed.')
    text = text.replace(old_lexical, new_lexical, 1)

    old_search_header = '''# ============================================================
# Search helpers
# ============================================================


def definition_search(query: str, top_k: int, books=None):
'''

    new_search_header = '''# ============================================================
# Search helpers
# ============================================================


def is_book_document(doc) -> bool:
    return doc.metadata.get("source_class", "book") == "book"


def is_official_document(doc) -> bool:
    return doc.metadata.get("source_class") == "official_correction"


def official_document_key(doc):
    return (
        doc.metadata.get("chunk_id"),
        doc.metadata.get("source_key"),
        doc.metadata.get("record_index"),
    )


def official_lexical_search(query: str, top_k: int):
    if (
        not official_documents
        or official_lexical_vectorizer is None
        or official_lexical_matrix is None
    ):
        return []

    query_vector = official_lexical_vectorizer.transform([query])
    scores = cosine_similarity(
        query_vector,
        official_lexical_matrix,
    ).flatten()
    sorted_indices = scores.argsort()[::-1]
    results = []

    for index in sorted_indices:
        score = float(scores[index])
        if score <= 0:
            break
        results.append((official_documents[index], score))
        if len(results) >= top_k:
            break

    return results


def official_search(
    query: str,
    top_k: int = 20,
    candidate_k: int = 100,
    variants: Optional[list[str]] = None,
):
    """公式エラッタだけをvector + lexicalのRRFで検索する。

    現段階では結果を /ask context へは入れない。
    次段階のOfficial Override Resolver用の独立検索口。
    """
    if not official_documents:
        return []

    query_variants = variants or build_query_variants(query)
    scores_by_key = defaultdict(float)
    docs_by_key = {}
    reasons_by_key = defaultdict(list)
    rrf_k = 60.0
    vector_weight = 0.40
    lexical_weight = 0.60

    for search_query in query_variants:
        # FAISS自体はbook/official共通なので、取得後にofficialだけへ絞る。
        # officialが601件と少ないため、book検索より広めに候補を取得する。
        vector_results = db.similarity_search_with_score(
            search_query,
            k=max(candidate_k, 200),
        )
        vector_position = 0
        for doc, _distance in vector_results:
            if not is_official_document(doc):
                continue
            vector_position += 1
            key = official_document_key(doc)
            docs_by_key[key] = doc
            scores_by_key[key] += vector_weight / (rrf_k + vector_position)
            if vector_position <= 15:
                reasons_by_key[key].append(
                    f"公式ベクトル検索 #{vector_position}（{search_query}）"
                )

        lexical_results = official_lexical_search(
            search_query,
            candidate_k,
        )
        for lexical_position, (doc, _score) in enumerate(
            lexical_results,
            start=1,
        ):
            key = official_document_key(doc)
            docs_by_key[key] = doc
            scores_by_key[key] += lexical_weight / (rrf_k + lexical_position)
            if lexical_position <= 15:
                reasons_by_key[key].append(
                    f"公式文字列検索 #{lexical_position}（{search_query}）"
                )

    scored = []
    for key, score in scores_by_key.items():
        doc = docs_by_key[key]
        scored.append((doc, score, reasons_by_key[key]))

    scored.sort(key=lambda item: item[1], reverse=True)

    result = []
    for doc, score, reasons in scored[:top_k]:
        result.append(
            {
                "doc": doc,
                "retrieval_score": score,
                "reason": "公式エラッタ検索（" + " / ".join(reasons[:3]) + "）",
            }
        )
    return result


def definition_search(query: str, top_k: int, books=None):
'''

    if old_search_header not in text:
        raise RuntimeError('Search helpers header was not found. qa_api.py may have changed.')
    text = text.replace(old_search_header, new_search_header, 1)

    old_vector_filter = '''        for doc, _distance in vector_results:
            if get_logical_page(doc) is None:
                continue
            if books and doc.metadata.get("book") not in books:
                continue
'''

    new_vector_filter = '''        for doc, _distance in vector_results:
            # 通常Hybridは書籍本文専用。公式エラッタはofficial_search()で別取得する。
            if not is_book_document(doc):
                continue
            if get_logical_page(doc) is None:
                continue
            if books and doc.metadata.get("book") not in books:
                continue
'''

    if old_vector_filter not in text:
        raise RuntimeError('Hybrid vector filter block was not found. qa_api.py may have changed.')
    text = text.replace(old_vector_filter, new_vector_filter, 1)

    dst.write_text(text, encoding='utf-8', newline='\n')
    compile(text, str(dst), 'exec')
    print(dst)


if __name__ == '__main__':
    main()
