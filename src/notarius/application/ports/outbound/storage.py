import abc
import pathlib
from collections.abc import Iterator
from contextlib import contextmanager

from notarius.domain.protocols import FileStreamProtocol


class FileStorage(abc.ABC):
    storage_root: pathlib.Path

    @abc.abstractmethod
    def save(self, stream: FileStreamProtocol, file_path: pathlib.Path) -> pathlib.Path:
        raise NotImplementedError

    @abc.abstractmethod
    @contextmanager
    def load(self, file_path: pathlib.Path) -> Iterator[FileStreamProtocol]:
        raise NotImplementedError

    @abc.abstractmethod
    def delete(self, file_path: pathlib.Path) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def exists(self, file_path: pathlib.Path) -> bool:
        raise NotImplementedError


class AbstractFileRepository[T](abc.ABC):
    """
    Abstract base class for a file repository.

    This class defines the blueprint for a file repository that interacts with a
    storage system. It provides an abstraction for adding and retrieving files
    from the underlying storage, ensuring a consistent interface for file
    management.

    Attributes:
        storage: A file storage instance used for managing file operations.
    """

    storage: FileStorage

    @abc.abstractmethod
    def add(self, file: T, name: str) -> pathlib.Path:
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, path: pathlib.Path) -> T:
        raise NotImplementedError

    @abc.abstractmethod
    def exists(self, name: str) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def get_path(self, name: str) -> pathlib.Path:
        raise NotImplementedError
