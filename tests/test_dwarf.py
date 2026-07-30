import unittest
from profila._dwarf import JITDWARFResolver, LineMapping


class TestJITDWARFResolver(unittest.TestCase):
    def test_jit_dwarf_resolver_empty(self) -> None:
        resolver = JITDWARFResolver()
        self.assertIsNone(resolver.resolve_pc(0x1000))

    def test_jit_dwarf_resolver_manual_mapping(self) -> None:
        resolver = JITDWARFResolver()
        resolver.mappings.append(
            LineMapping(
                start_address=0x1000,
                end_address=0x1050,
                filename="/tmp/test_script.py",
                line=42,
            )
        )
        
        frame = resolver.resolve_pc(0x1020)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(frame.file, "/tmp/test_script.py")
        self.assertEqual(frame.line, 42)

        self.assertIsNone(resolver.resolve_pc(0x1050))
        self.assertIsNone(resolver.resolve_pc(0x0F00))

    def test_jit_dwarf_resolver_invalid_elf(self) -> None:
        resolver = JITDWARFResolver()
        count = resolver.add_elf_image(b"not an elf file")
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
