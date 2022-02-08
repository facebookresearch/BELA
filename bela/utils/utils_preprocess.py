import bs4
from bs4 import BeautifulSoup
import html
import json
from urllib.parse import unquote
import xml.etree.ElementTree as ET
from nltk.tokenize import sent_tokenize


def search_wikidata(query, label_alias2wikidata_id):
    return list(set(label_alias2wikidata_id.get(query.lower(), [])))


def search_wikipedia(title, lang, lang_title2wikidataID, lang_redirect2title):

    max_redirects = 10
    while (lang, title) in lang_redirect2title and max_redirects > 0:

        title = lang_redirect2title[(lang, title)]
        max_redirects -= 1

    if (lang, title) in lang_title2wikidataID:
        return True, lang_title2wikidataID[(lang, title)]
    else:
        return False, title


def search_simple(anchor, lang, lang_title2wikidataID):
    if "http" in anchor:
        return True, []

    unquoted = html.unescape(unquote(anchor).split("#")[0].replace("_", " "))
    if unquoted[0:2] == "w:":
        unquoted = unquoted[2:]
    if unquoted[0:3] == ":w:":
        unquoted = unquoted[3:]
    if unquoted == "":
        return True, []

    unquoted = unquoted[0].upper() + unquoted[1:]
    if (lang, unquoted) in lang_title2wikidataID:
        return True, lang_title2wikidataID[(lang, unquoted)]
    else:
        return False, unquoted


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


def chunk_it(seq, num):
    assert num > 0
    chunk_len = len(seq) // num
    chunks = [seq[i * chunk_len: i * chunk_len + chunk_len] for i in range(num)]

    diff = len(seq) - chunk_len * num
    for i in range(diff):
        chunks[i].append(seq[chunk_len * num + i])

    return chunks


def extract_pages(filename):
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

def sort_time(path):
    time_dict = {}
    with open(path) as f:
        for line in f:
            line = json.loads(line)
            time = line["time_stamp"]
            if time not in time_dict:
                time_dict[time] = [line]
            else:
                time_dict[time].append(line)
    with open(path.split(".")[0] + "_sorted.jsonl", "w") as f:
        for key in sorted(time_dict):
            for line in time_dict[key]:
                f.write(json.dumps(time_dict[key]))
                f.write("\n")

def split_paragraph_max_seq_length(text, f_out, tokenizer, idx, seq_length=256):
    current_paragraph = ""
    current_length = 0
    sentences = sent_tokenize(text['text'])
    num = 0
    for sentence in sentences:
        if sentence=="":
            continue
        sentence_tokenized = tokenizer.tokenize(sentence)
        if len(sentence_tokenized) > seq_length:
            continue
        if current_length + len(sentence_tokenized) <= seq_length:
            current_length += len(sentence_tokenized)
            current_paragraph += sentence
            current_paragraph += " "
        else:
            data = {"text": current_paragraph.strip(), "id": idx + "_" + str(num), 'time_stamp': text['time_stamp']}
            f_out.write(json.dumps(data))
            f_out.write("\n")
            current_paragraph = ""
            current_length = 0
            current_length += len(sentence_tokenized)
            current_paragraph += sentence
            current_paragraph += " "
            num += 1
    if len(current_paragraph)!=0:
        data = {"text": current_paragraph.strip(), "id": idx + "_" + str(num), 'time_stamp': text['time_stamp']}
        f_out.write(json.dumps(data))
        f_out.write("\n")
