from typing import List, Optional

import fitz
from fitz import (
    Document as FitzDocument,
    Page
)

from .base import BaseParser


class PymuPDFParser(BaseParser):
    def __init__(self,
                 save_chunks: bool = False,
                 save_location: Optional[str] = None,
                 header_height: int = 125,
                 **kwargs):
        super().__init__(save_chunks, save_location)
        self.header_height = header_height

    def _parse_file(self, file_path: str, **kwargs) -> List[str]:
        doc = fitz.open(file_path)
        new_docs = self._remove_header(doc)
        titles_index = self._get_title_index(new_docs)
        raw_chunks = self._extract_raw_chunks(new_docs, titles_index)
        return raw_chunks

    def _remove_header(self, doc: FitzDocument) -> List[FitzDocument]:
        new_docs = []
        for page_num in range(len(doc)):
            new_doc = fitz.open()
            page = doc.load_page(page_num)
            rect = page.rect
            crop_rect = fitz.Rect(rect.x0, rect.y0 + self.header_height, rect.x1, rect.y1)

            new_page = new_doc.new_page(width=crop_rect.width, height=crop_rect.height)
            new_page.show_pdf_page(new_page.rect, doc, page_num, clip=crop_rect)

            new_docs.append(new_doc)

        return new_docs

    def _get_title_index(self, docs: List[FitzDocument]):
        titles_index = []
        for idx, doc in enumerate(docs):
            doc.save(f"output_{idx}.pdf")
            title_idx_in_page = []
            page = doc.load_page(0)
            blocks = page.get_text("dict")["blocks"]
            for num_block in range(len(blocks)):
                if blocks[num_block]["type"] == 0:
                    for line in blocks[num_block]["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text.isupper() and len(text) > 1:
                                if len(title_idx_in_page) == 0 and blocks[num_block]["number"] > 0:
                                    title_idx_in_page.append(0)
                                    title_idx_in_page.append(blocks[num_block]["number"])
                                elif len(title_idx_in_page) > 0:
                                    title_idx_in_page.append(blocks[num_block]["number"])

            if len(title_idx_in_page) > 0:
                title_idx_in_page.append(blocks[-1]["number"])
            elif len(title_idx_in_page) == 0:
                title_idx_in_page.append(0)
                title_idx_in_page.append(blocks[-1]["number"])

            titles_index.append(title_idx_in_page)

        titles_index = [[lst[i] for i in range(len(lst)) if i == 0 or lst[i] != lst[i-1]] for lst in titles_index]

        return titles_index

    def _extract_raw_chunks(self, docs: List[FitzDocument], titles_index: List[List[int]]) -> List[str]:

        joined_texts = []
        for idx, doc in enumerate(docs):
            page = doc.load_page(0)
            blocks = page.get_text("dict")["blocks"]

            index = titles_index[idx]
            pairs = list(zip(index, index[1:]))
            for pair_idx in range(len(pairs)):

                start_block = pairs[pair_idx][0]
                end_block = pairs[pair_idx][1]

                spans_text = []
                for block in blocks:

                    if start_block <= block['number'] <= end_block:
                        for line in block['lines']:
                            for span in line['spans']:
                                spans_text.append((span['text'], span['bbox']))

                joined_text = "\n".join([span[0] for span in spans_text])
                joined_text = joined_text.strip()
                if len(joined_text) <= 1:
                    continue

                # joined_text = f"{title_name}\n{joined_text}"
                joined_texts.append(joined_text)

        return joined_texts
