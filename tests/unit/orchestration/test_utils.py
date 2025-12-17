from notarius.orchestration.utils import AssetKeyHelper


class TestAssetKeyHelper:
    """Tests for AssetKeyHelper utility class."""

    def test_build_key_single_part(self):
        """Build asset key from a single part."""
        key = AssetKeyHelper.build_key("part1")
        assert key == "part1"

    def test_build_key_multiple_parts(self):
        """Build asset key from multiple parts."""
        key = AssetKeyHelper.build_key("part1", "part2", "part3")
        assert key == "part1__part2__part3"

    def test_build_key_empty_parts(self):
        """Build asset key with no parts returns empty string."""
        key = AssetKeyHelper.build_key()
        assert key == ""

    def test_build_prefixed_key(self):
        """Build asset key with layer and source prefix."""
        key = AssetKeyHelper.build_prefixed_key("raw", "source1", "table1")
        assert key == "raw__source1__table1"

    def test_build_prefixed_key_multiple_parts(self):
        """Build prefixed asset key with multiple additional parts."""
        key = AssetKeyHelper.build_prefixed_key(
            "processed", "source2", "part1", "part2", "part3"
        )
        assert key == "processed__source2__part1__part2__part3"

    def test_build_prefixed_key_no_additional_parts(self):
        """Build prefixed asset key with only layer and source."""
        key = AssetKeyHelper.build_prefixed_key("layer", "source")
        assert key == "layer__source__"

    def test_build_key_with_special_characters(self):
        """Build asset key with parts containing special characters."""
        key = AssetKeyHelper.build_key("part-1", "part_2", "part.3")
        assert key == "part-1__part_2__part.3"

    def test_build_prefixed_key_with_special_characters(self):
        """Build prefixed asset key with special characters in parts."""
        key = AssetKeyHelper.build_prefixed_key("my-layer", "my_source", "my.part")
        assert key == "my-layer__my_source__my.part"
