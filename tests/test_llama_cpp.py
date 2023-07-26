from sqlite_utils.plugins import get_plugins


def test_plugin_is_installed():
    plugins = get_plugins()
    names = [plugin["name"] for plugin in plugins]
    assert "sqlite-utils-llama-cpp" in names
