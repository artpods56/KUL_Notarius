class ConfigFileNotFoundError(Exception):
    """Raise when specified config_manager file is not found."""

    pass


class ConfigNotRegisteredError(Exception):
    """Raise when specified config_manager is not registered."""

    pass


class InvalidConfigType(Exception):
    """Raise when specified config_manager type is invalid."""

    pass


class InvalidConfigSubtype(Exception):
    """Raise when specified config_manager subtype is invalid."""

    pass
