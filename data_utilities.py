import os

FAKE_TEXT = "data/text/fake"
TRUE_TEXT = "data/text/true"
FAKE_DOC = "data/doc/norm_fake"
TRUE_DOC = "data/doc/norm_true"
FAKE_SENT = "data/sent/norm_fake"
TRUE_SENT = "data/sent/norm_true"
FAKE_FILE = ["train.txt", "dev.txt", "test.txt"]
TRUE_FILE = ["0_true.txt", "1_true.txt", "2_true.txt", "3_true.txt", "4_true.txt",
             "5_true.txt", "6_true.txt", "7_true.txt", "8_true.txt", "9_true.txt"]


def load_doc(file_name):
    with open(file_name) as f:
        lines = f.readlines()
    return [l.strip() for l in lines]


def load_sent(file_name):
    docs = []
    doc = []
    with open(file_name) as f:
        lines = f.readlines()
    for line in lines:
        if line == "******\n":
            docs.append(doc)
            doc = []
        else:
            doc.append(line.strip())
    return docs


def load_feature_set(text_path, sent_path, doc_path, label, doc=False, sent_ling=True, doc_ling=True):
    text = load_sent(text_path)
    if doc:
        text = list2doc(text)
    sent_feature = load_sent(sent_path)
    doc_features = load_doc(doc_path)
    assert len(text) == len(sent_feature) == len(doc_features)
    labels = [label for _ in range(len(text))]
    if sent_ling and doc_ling:
        return list(zip(text, sent_feature, doc_features, labels))
    elif sent_ling and not doc_ling:
        return list(zip(text, sent_feature, labels))
    elif not sent_ling and doc_ling:
        return list(zip(text, doc_features, labels))
    elif not sent_ling and not doc_ling:
        return list(zip(text, labels))


def load_true(doc=False, sent_ling=True, doc_ling=True):
    train = []
    dev = []
    test = []
    for i in range(6):
        file_name = TRUE_FILE[i]
        text_path = os.path.join(TRUE_TEXT, file_name)
        sent_path = os.path.join(TRUE_SENT, file_name)
        doc_path = os.path.join(TRUE_DOC, file_name)
        train.append(load_feature_set(text_path, sent_path, doc_path, label=1,
                                      doc=doc, sent_ling=sent_ling, doc_ling=doc_ling))
    for i in range(6, 8):
        file_name = TRUE_FILE[i]
        text_path = os.path.join(TRUE_TEXT, file_name)
        sent_path = os.path.join(TRUE_SENT, file_name)
        doc_path = os.path.join(TRUE_DOC, file_name)
        dev += load_feature_set(text_path, sent_path, doc_path, label=1,
                                doc=doc, sent_ling=sent_ling, doc_ling=doc_ling)
    for i in range(8, 10):
        file_name = TRUE_FILE[i]
        text_path = os.path.join(TRUE_TEXT, file_name)
        sent_path = os.path.join(TRUE_SENT, file_name)
        doc_path = os.path.join(TRUE_DOC, file_name)
        test += load_feature_set(text_path, sent_path, doc_path, label=1,
                                 doc=doc, sent_ling=sent_ling, doc_ling=doc_ling)
    return train, dev, test


def load_fake(doc=False, sent_ling=True, doc_ling=True):
    fake = []
    for file_name in FAKE_FILE:
        text_path = os.path.join(FAKE_TEXT, file_name)
        sent_path = os.path.join(FAKE_SENT, file_name)
        doc_path = os.path.join(FAKE_DOC, file_name)
        features = load_feature_set(text_path, sent_path, doc_path, label=0,
                                    doc=doc, sent_ling=sent_ling, doc_ling=doc_ling)
        fake.append(features)
    return fake[0], fake[1], fake[2]


def list2doc(docs):
    """
        convert a list of sentences to a single document
        """
    doc_docs = []
    s = ''
    for doc in docs:
        if isinstance(doc, tuple):
            doc = doc[0]
        else:
            doc = doc
        for sent in doc:
            s += sent + ' '
        doc_docs.append(s)
        s = ''
    return doc_docs


def list2file(docs, file):
    with open(file, 'a') as f:
        for doc in docs:
            f.write(doc[0])
            f.write('\n')


if __name__ == '__main__':
    print()
    train, dev, test = load_fake(True, False, False)
    print(len(train), len(dev), len(test))
    list2file(train, "fake_doc_train.txt")
    list2file(dev, "fake_doc_dev.txt")
    list2file(test, "fake_doc_test.txt")

