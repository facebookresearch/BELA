import xml.etree.ElementTree as ET
import bs4
from bs4 import BeautifulSoup
import html

class DummyPathManager:
    def get_local_path(self, path, *args, **kwargs):
        return path

    def open(self, path, *args, **kwargs):
        return open(path, *args, **kwargs)


PathManager = DummyPathManager()


def search_wikidata(query, label_alias2wikidataID):
    return list(set(label_alias2wikidataID.get(query.lower(), [])))


def get_wikidata_ids(
    anchor,
    lang,
    lang_title2wikidataID,
    lang_redirect2title,
    label_or_alias2wikidataID,
):
    success, result = search_simple(anchor, lang, label_or_alias2wikidataID)
    if success:
        return result, "simple"
    else:
        success, result = search_wikipedia(
            result, lang, lang_title2wikidataID, lang_redirect2title
        )
        if success:
            return result, "wikipedia"
        else:
            return search_wikidata(result, label_or_alias2wikidataID), "wikidata"


def extract_pages(filename):
    print('filename', filename)
    docs = {}
    with open(filename) as f:
        for line in f:
            # CASE 1: beginning of the document
            if line.startswith("<doc id="):
                doc = ET.fromstring("{}{}".format(line, "</doc>")).attrib
                doc["paragraphs"] = []
                doc["anchors"] = []

            # CASE 2: end of the document
            elif line.startswith("</doc>"):
                assert doc["id"] not in docs, "{} ({}) already in dict as {}".format(
                    doc["id"], doc["title"], docs[doc["id"]]["title"]
                )
                docs[doc["id"]] = doc

            # CASE 3: in the document
            else:
                doc["paragraphs"].append("")
                line = html.unescape(line)
                try:
                    line = BeautifulSoup(line, "html.parser")
                except:
                    print("error line `{}`".format(line))
                    line = [line]

                for span in line:
                    if isinstance(span, bs4.element.Tag):
                        if span.get("href", None):
                            doc["anchors"].append(
                                {
                                    "text": span.get_text(),
                                    "href": span["href"],
                                    "paragraph_id": len(doc["paragraphs"]) - 1,
                                    "start": len(doc["paragraphs"][-1]),
                                    "end": len(doc["paragraphs"][-1])
                                    + len(span.get_text()),
                                }
                            )
                        doc["paragraphs"][-1] += span.get_text()
                    else:
                        doc["paragraphs"][-1] += str(span)

    return docs


def chunk_it(seq, num):
    assert num > 0
    chunk_len = len(seq) // num
    chunks = [seq[i * chunk_len : i * chunk_len + chunk_len] for i in range(num)]

    diff = len(seq) - chunk_len * num
    for i in range(diff):
        chunks[i].append(seq[chunk_len * num + i])

    return chunks
