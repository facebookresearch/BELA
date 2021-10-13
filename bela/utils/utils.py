class DummyPathManager:
    def get_local_path(self, path, *args, **kwargs):
        return path

    def open(self, path, *args, **kwargs):
        return open(path, *args, **kwargs)


PathManager = DummyPathManager()

def embed_novel_entites():
    pass

def add_novel_entities2faiss(path2entities):
    with open(path2entities)


