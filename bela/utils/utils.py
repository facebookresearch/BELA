

class DummyPathManager:
    def get_local_path(self, path, *args, **kwargs):
        return path

    def open(self, path, *args, **kwargs):
        return open(path, *args, **kwargs)


PathManager = DummyPathManager()