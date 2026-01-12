class Singleton:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        elif args or kwargs:
            raise ValueError(
                f"Singleton {cls.__name__} is already initialized; do not pass arguments again"
            )
        return cls._instance

    def __init__(self, *args, **kwargs):
        if self.__class__._initialized:
            return
        self._singleton_init(*args, **kwargs)
        self.__class__._initialized = True

    def _singleton_init(self, *args, **kwargs):
        """To be implemented by subclass."""
        raise NotImplementedError("Subclasses must implement _singleton_init()")

