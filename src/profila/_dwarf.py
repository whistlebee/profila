"""
Direct LLVM DWARF Symbol Unwinder and Resolver for Numba JIT compiled code.

Numba JIT registers dynamically generated object files with debug information
via the standard GDB JIT Interface (`__jit_debug_descriptor`). This module
parses those in-memory ELF objects and reads DWARF `.debug_line` tables using
pyelftools to map instruction pointers directly to source files and lines.
"""

from dataclasses import dataclass
import io
import struct
from typing import Callable, Dict, List, Optional, Tuple

from elftools.elf.elffile import ELFFile
from elftools.dwarf.dwarfinfo import DWARFInfo
from elftools.dwarf.lineprogram import LineProgram, LineState

from ._stats import Frame


@dataclass
class LineMapping:
    """Range of instructions mapped to a source file and line."""
    start_address: int
    end_address: int
    filename: str
    line: int


class JITDWARFResolver:
    """
    Parses LLVM DWARF debug information from Numba JIT ELF images.
    """

    def __init__(self) -> None:
        self.mappings: List[LineMapping] = []
        self._parsed_addrs: set[int] = set()

    def add_elf_image(self, elf_bytes: bytes) -> int:
        """
        Parse an ELF image from bytes and index its DWARF line tables.
        Returns the number of line mappings extracted.
        """
        try:
            stream = io.BytesIO(elf_bytes)
            elffile = ELFFile(stream)
            if not elffile.has_dwarf_info():
                return 0

            dwarf_info: DWARFInfo = elffile.get_dwarf_info()
            count = 0
            for cu in dwarf_info.iter_CUs():
                line_program: Optional[LineProgram] = dwarf_info.line_program_for_CU(cu)
                if line_program is None:
                    continue

                entries = line_program.get_entries()
                prev_state: Optional[LineState] = None

                for entry in entries:
                    state = entry.state
                    if state is None:
                        continue

                    if prev_state is not None and prev_state.file > 0 and not prev_state.end_sequence:
                        filename_bytes = line_program.header["file_entry"][prev_state.file - 1].name
                        filename = filename_bytes.decode("utf-8", errors="replace")
                        
                        start_addr = prev_state.address
                        end_addr = state.address

                        if end_addr > start_addr and prev_state.line is not None:
                            mapping = LineMapping(
                                start_address=start_addr,
                                end_address=end_addr,
                                filename=filename,
                                line=prev_state.line,
                            )
                            self.mappings.append(mapping)
                            count += 1

                    if state.end_sequence:
                        prev_state = None
                    else:
                        prev_state = state

            # Sort mappings by start address for efficient binary searching
            self.mappings.sort(key=lambda m: m.start_address)
            return count
        except Exception:
            return 0

    def resolve_pc(self, pc: int) -> Optional[Frame]:
        """
        Resolve an instruction pointer (PC) to a Frame(file, line).
        """
        for mapping in self.mappings:
            if mapping.start_address <= pc < mapping.end_address:
                return Frame(file=mapping.filename, line=mapping.line)
        return None

    def parse_jit_descriptor_memory(
        self,
        descriptor_addr: int,
        read_memory: Callable[[int, int], bytes],
        is_64bit: bool = True,
    ) -> int:
        """
        Traverse the __jit_debug_descriptor linked list from process memory
        and load all registered ELF symbol files.

        struct jit_descriptor {
          uint32_t version;
          uint32_t action_flag;
          jit_code_entry *relevant_entry;
          jit_code_entry *first_entry;
        };

        struct jit_code_entry {
          jit_code_entry *next_entry;
          jit_code_entry *prev_entry;
          const char *symfile_addr;
          uint64_t symfile_size;
        };
        """
        ptr_format = "<Q" if is_64bit else "<I"
        ptr_size = 8 if is_64bit else 4

        # Read jit_descriptor
        desc_bytes = read_memory(descriptor_addr, 8 + 2 * ptr_size)
        version, action_flag = struct.unpack("<II", desc_bytes[:8])
        if version != 1:
            return 0

        first_entry_addr = struct.unpack(ptr_format, desc_bytes[8 + ptr_size:8 + 2 * ptr_size])[0]
        current_entry = first_entry_addr
        total_mappings = 0

        while current_entry != 0:
            entry_bytes = read_memory(current_entry, 2 * ptr_size + ptr_size + 8)
            next_entry = struct.unpack(ptr_format, entry_bytes[:ptr_size])[0]
            symfile_addr = struct.unpack(ptr_format, entry_bytes[2 * ptr_size:3 * ptr_size])[0]
            symfile_size = struct.unpack("<Q", entry_bytes[3 * ptr_size:3 * ptr_size + 8])[0]

            if symfile_addr != 0 and symfile_size > 0 and symfile_addr not in self._parsed_addrs:
                self._parsed_addrs.add(symfile_addr)
                elf_data = read_memory(symfile_addr, symfile_size)
                total_mappings += self.add_elf_image(elf_data)

            current_entry = next_entry

        return total_mappings
