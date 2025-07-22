# Copyright (c) 2023 Anthropic
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 13 June 2025
#
# Original file was released under MIT License, with the full license text
# available at https://github.com/anthropics/anthropic-quickstarts/blob/main/LICENSE
#
# This modified file is released under the same license.

from pathlib import Path
from typing import override

from .base import Tool, ToolError, ToolExecResult, ToolParameter, ToolCallArguments
from .run import maybe_truncate, run

EditToolSubCommands = [
    "view",
    "create",
    "str_replace",
    "insert",
]
SNIPPET_LINES: int = 4


class TextEditorTool(Tool):
    """Tool to replace a string in a file."""

    @override
    def get_name(self) -> str:
        return "str_replace_based_edit_tool"

    @override
    def get_description(self) -> str:
        return """Custom editing tool for viewing, creating and editing files
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file !!! If you know that the `path` already exists, please remove it first and then perform the `create` operation!
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`
"""

    @override
    def get_parameters(self) -> list[ToolParameter]:
        """Get the parameters for the str_replace_based_edit_tool."""
        return [
            ToolParameter(
                name="command",
                type="string",
                description=f"The commands to run. Allowed options are: {', '.join(EditToolSubCommands)}.",
                required=True,
                enum=EditToolSubCommands,
            ),
            ToolParameter(
                name="path",
                type="string",
                description="Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`.",
                required=True,
            ),
            ToolParameter(
                name="file_text",
                type="string",
                description="Required parameter of `create` command, with the content of the file to be created.",
                required=False,
            ),
            ToolParameter(
                name="insert_line",
                type="integer",
                description="Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.",
                required=False,
            ),
            ToolParameter(
                name="new_str",
                type="string",
                description="Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.",
                required=False,
            ),
            ToolParameter(
                name="old_str",
                type="string",
                description="Required parameter of `str_replace` command containing the string in `path` to replace.",
                required=False,
            ),
            ToolParameter(
                name="view_range",
                type="array",
                description="Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.",
                items={"type": "integer"},
                required=False,
            ),
        ]

    @override
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        """Execute the str_replace_editor tool."""
        command = str(arguments["command"]) if "command" in arguments else None
        if command is None:
            return ToolExecResult(
                error=f"No command provided for the {self.get_name()} tool",
                error_code=-1,
            )
        path = str(arguments["path"]) if "path" in arguments else None
        if path is None:
            return ToolExecResult(
                error=f"No path provided for the {self.get_name()} tool", error_code=-1
            )
        _path = Path(path)
        
        # Handle path conversion and validation
        try:
            validated_path = self.validate_and_convert_path(command, _path)
        except ToolError as e:
            return ToolExecResult(error=str(e), error_code=-1)

        try:
            if command == "view":
                view_range = arguments.get("view_range", None)
                return await self.view(validated_path, view_range)  # pyright: ignore[reportArgumentType]
            elif command == "create":
                file_text = arguments.get("file_text", None)
                if file_text is None:
                    return ToolExecResult(
                        error="Parameter `file_text` is required for command: create",
                        error_code=-1,
                    )
                self.write_file(validated_path, file_text)  # pyright: ignore[reportArgumentType]
                return ToolExecResult(output=f"File created successfully at: {validated_path}")
            elif command == "str_replace":
                old_str = arguments.get("old_str") if "old_str" in arguments else None
                if old_str is None:
                    return ToolExecResult(
                        error="Parameter `old_str` is required for command: str_replace",
                        error_code=-1,
                    )
                new_str = arguments.get("new_str") if "new_str" in arguments else None
                return self.str_replace(validated_path, old_str, new_str)  # pyright: ignore[reportArgumentType]
            elif command == "insert":
                insert_line = (
                    arguments.get("insert_line") if "insert_line" in arguments else None
                )
                if insert_line is None:
                    return ToolExecResult(
                        error="Parameter `insert_line` is required for command: insert",
                        error_code=-1,
                    )
                new_str_to_insert = (
                    arguments.get("new_str") if "new_str" in arguments else None
                )
                if new_str_to_insert is None:
                    return ToolExecResult(
                        error="Parameter `new_str` is required for command: insert",
                        error_code=-1,
                    )
                return self.insert(validated_path, insert_line, new_str_to_insert)  # pyright: ignore[reportArgumentType]
            else:
                return ToolExecResult(
                    error=f"Unrecognized command {command}. The allowed commands for the {self.name} tool are: {', '.join(EditToolSubCommands)}",
                    error_code=-1,
                )
        except ToolError as e:
            return ToolExecResult(error=str(e), error_code=-1)

    def validate_and_convert_path(self, command: str, path: Path) -> Path:
        """Validate the path for the str_replace_editor tool and return the corrected path."""
        import os
        import sys
        
        # Convert path to string for easier processing
        path_str = str(path)
        
        # Print debug info
        print(f"Original input path: {path_str}", file=sys.stderr)
        print(f"Working directory: {os.getcwd()}", file=sys.stderr)
        
        # Handle Git Bash style paths on Windows
        if os.name == 'nt' and (path_str.startswith('/') or path_str.startswith('\\')):
            # Remove leading slash for path processing
            clean_path = path_str[1:] if path_str.startswith('/') or path_str.startswith('\\') else path_str
            
            # Check if it's a Git Bash style path like /c/Users/...
            if len(clean_path) >= 2 and clean_path[0].isalpha() and (clean_path[1] == '/' or clean_path[1] == '\\'):
                # Convert /c/Users/... to C:/Users/...
                drive_letter = clean_path[0].upper()
                # Get the rest of the path without the drive letter part
                if clean_path[1] == '/':
                    rest_of_path = clean_path[2:].replace('/', os.sep)
                else:
                    rest_of_path = clean_path[2:].replace('\\', os.sep)
                
                # Create the Windows-style path
                windows_path = f"{drive_letter}:{os.sep}{rest_of_path}"
                final_path = Path(windows_path)
                print(f"Converted Git Bash style path to: {final_path}", file=sys.stderr)
                
                # Check if this path exists
                if not final_path.exists() and command != "create":
                    # Try the path as-is from current directory
                    alt_path = Path(os.path.join(os.getcwd(), clean_path.replace('/', os.sep).replace('\\', os.sep)))
                    if alt_path.exists():
                        final_path = alt_path
                        print(f"Using alternative path: {final_path}", file=sys.stderr)
            else:
                # It's a relative path with Unix separators
                # Convert to Windows path by joining with current directory
                unix_path = clean_path.replace('/', os.sep).replace('\\', os.sep)
                final_path = Path(os.getcwd()) / unix_path
                print(f"Converted relative Unix path to: {final_path}", file=sys.stderr)
        elif not path.is_absolute():
            # Standard relative path
            final_path = Path.cwd() / path
            print(f"Converted relative path to: {final_path}", file=sys.stderr)
        else:
            # Path is already absolute
            final_path = path
            print(f"Using absolute path: {final_path}", file=sys.stderr)
        
        # For create command, we allow non-existent paths but parent directory must exist or be created
        if command == "create":
            # If the file already exists and we're trying to create it, return an error
            if final_path.exists() and final_path.is_file():
                raise ToolError(f"File already exists at: {final_path}. Cannot overwrite files using command `create`.")
                
            # If the path exists but is a directory, return an error
            if final_path.exists() and final_path.is_dir():
                raise ToolError(f"Path {final_path} is a directory. Cannot create a file with the same name.")
                
            # Ensure parent directory exists
            try:
                if not final_path.parent.exists():
                    final_path.parent.mkdir(parents=True, exist_ok=True)
                    print(f"Created parent directory: {final_path.parent}", file=sys.stderr)
            except Exception as e:
                raise ToolError(f"Failed to create parent directory for {final_path}: {e}")
                
            return final_path
        
        # For all other commands, verify the path exists
        if not final_path.exists():
            if os.path.exists(str(final_path)):
                # Path exists according to os.path but not Path.exists()
                print(f"Path exists according to os.path but not Path.exists(): {final_path}", file=sys.stderr)
                # Use string path instead
                final_path = Path(os.path.normpath(str(final_path)))
            else:
                # Try some alternative paths as a last resort
                alt_path1 = Path(os.path.normpath(os.path.join("c:", path_str)))
                alt_path2 = Path(os.path.normpath(os.path.join("c:", path_str[1:] if path_str.startswith('/') else path_str)))
                
                print(f"Trying alternative paths:", file=sys.stderr)
                print(f"  Alt1: {alt_path1} (exists: {alt_path1.exists()})", file=sys.stderr)
                print(f"  Alt2: {alt_path2} (exists: {alt_path2.exists()})", file=sys.stderr)
                
                if alt_path1.exists():
                    final_path = alt_path1
                    print(f"Using alternative path 1: {final_path}", file=sys.stderr)
                elif alt_path2.exists():
                    final_path = alt_path2
                    print(f"Using alternative path 2: {final_path}", file=sys.stderr)
                else:
                    # If we still can't find a valid path, raise an error
                    raise ToolError(f"The path {final_path} does not exist. Please provide a valid path.")
        
        # Check if the path points to a directory for non-view commands
        if final_path.is_dir() and command != "view":
            raise ToolError(f"The path {final_path} is a directory and only the `view` command can be used on directories")
        
        print(f"Final validated path: {final_path} (exists: {final_path.exists()})", file=sys.stderr)
        return final_path

    async def view(
        self, path: Path, view_range: list[int] | None = None
    ) -> ToolExecResult:
        """Implement the view command"""
        if path.is_dir():
            if view_range:
                raise ToolError(
                    "The `view_range` parameter is not allowed when `path` points to a directory."
                )

            # Double-check that the path actually exists
            if not path.exists():
                return ToolExecResult(
                    error=f"Error: Directory {path} does not exist", 
                    error_code=1
                )

            # Use Windows-compatible directory listing
            import os
            if os.name == 'nt':
                # Windows: use dir command or Python's os.walk
                try:
                    verified_files = []
                    # List only files and directories that actually exist
                    for item in path.iterdir():
                        if not item.name.startswith('.') and item.exists():  # Exclude hidden items and verify existence
                            if item.is_dir():
                                verified_files.append(f"{item.name}/")
                                # List one level deeper
                                try:
                                    for subitem in item.iterdir():
                                        if not subitem.name.startswith('.') and subitem.exists():
                                            if subitem.is_dir():
                                                verified_files.append(f"{item.name}/{subitem.name}/")
                                            else:
                                                verified_files.append(f"{item.name}/{subitem.name}")
                                except (PermissionError, OSError):
                                    continue
                            else:
                                verified_files.append(item.name)
                    
                    stdout = f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n"
                    stdout += "\n".join(verified_files) + "\n"
                    return ToolExecResult(error_code=0, output=stdout, error="")
                except Exception as e:
                    return ToolExecResult(error_code=1, output="", error=str(e))
            else:
                # Unix-like systems: use find command
                return_code, stdout, stderr = await run(
                    rf"find {path} -maxdepth 2 -not -path '*/\.*'"
                )
                if not stderr:
                    stdout = f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n{stdout}\n"
                return ToolExecResult(error_code=return_code, output=stdout, error=stderr)

        file_content = self.read_file(path)
        init_line = 1
        if view_range:
            if len(view_range) != 2 or not all(isinstance(i, int) for i in view_range):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise ToolError(
                    "Invalid `view_range`. It should be a list of two integers."
                )
            file_lines = file_content.split("\n")
            n_lines_file = len(file_lines)
            init_line, final_line = view_range
            if init_line < 1 or init_line > n_lines_file:
                raise ToolError(
                    f"Invalid `view_range`: {view_range}. Its first element `{init_line}` should be within the range of lines of the file: {[1, n_lines_file]}"
                )
            if final_line > n_lines_file:
                raise ToolError(
                    f"Invalid `view_range`: {view_range}. Its second element `{final_line}` should be smaller than the number of lines in the file: `{n_lines_file}`"
                )
            if final_line != -1 and final_line < init_line:
                raise ToolError(
                    f"Invalid `view_range`: {view_range}. Its second element `{final_line}` should be larger or equal than its first `{init_line}`"
                )

            if final_line == -1:
                file_content = "\n".join(file_lines[init_line - 1 :])
            else:
                file_content = "\n".join(file_lines[init_line - 1 : final_line])

        return ToolExecResult(
            output=self._make_output(file_content, str(path), init_line=init_line)
        )

    def str_replace(
        self, path: Path, old_str: str, new_str: str | None
    ) -> ToolExecResult:
        """Implement the str_replace command, which replaces old_str with new_str in the file content"""
        # Read the file content
        file_content = self.read_file(path).expandtabs()
        old_str = old_str.expandtabs()
        new_str = new_str.expandtabs() if new_str is not None else ""

        # Check if old_str is empty
        if not old_str:
            raise ToolError(
                f"No replacement was performed, old_str cannot be empty."
            )

        # Check if old_str is unique in the file
        occurrences = file_content.count(old_str)
        if occurrences == 0:
            raise ToolError(
                f"No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}."
            )
        elif occurrences > 1:
            file_content_lines = file_content.split("\n")
            lines = [
                idx + 1
                for idx, line in enumerate(file_content_lines)
                if old_str in line
            ]
            raise ToolError(
                f"No replacement was performed. Multiple occurrences of old_str `{old_str}` in lines {lines}. Please ensure it is unique"
            )

        # Replace old_str with new_str
        new_file_content = file_content.replace(old_str, new_str)

        # Write the new content to the file
        self.write_file(path, new_file_content)

        # Create a snippet of the edited section
        replacement_line = file_content.split(old_str)[0].count("\n")
        start_line = max(0, replacement_line - SNIPPET_LINES)
        end_line = replacement_line + SNIPPET_LINES + new_str.count("\n")
        snippet = "\n".join(new_file_content.split("\n")[start_line : end_line + 1])

        # Prepare the success message
        success_msg = f"The file {path} has been edited. "
        success_msg += self._make_output(
            snippet, f"a snippet of {path}", start_line + 1
        )
        success_msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."

        return ToolExecResult(
            output=success_msg,
        )

    def insert(self, path: Path, insert_line: int, new_str: str) -> ToolExecResult:
        """Implement the insert command, which inserts new_str at the specified line in the file content."""
        file_text = self.read_file(path).expandtabs()
        new_str = new_str.expandtabs()
        file_text_lines = file_text.split("\n")
        n_lines_file = len(file_text_lines)

        if insert_line < 0 or insert_line > n_lines_file:
            raise ToolError(
                f"Invalid `insert_line` parameter: {insert_line}. It should be within the range of lines of the file: {[0, n_lines_file]}"
            )

        new_str_lines = new_str.split("\n")
        new_file_text_lines = (
            file_text_lines[:insert_line]
            + new_str_lines
            + file_text_lines[insert_line:]
        )
        snippet_lines = (
            file_text_lines[max(0, insert_line - SNIPPET_LINES) : insert_line]
            + new_str_lines
            + file_text_lines[insert_line : insert_line + SNIPPET_LINES]
        )

        new_file_text = "\n".join(new_file_text_lines)
        snippet = "\n".join(snippet_lines)

        self.write_file(path, new_file_text)

        success_msg = f"The file {path} has been edited. "
        success_msg += self._make_output(
            snippet,
            "a snippet of the edited file",
            max(1, insert_line - SNIPPET_LINES + 1),
        )
        success_msg += "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary."
        return ToolExecResult(
            output=success_msg,
        )

    # Note: undo_edit method is not implemented in this version as it was removed

    def read_file(self, path: Path):
        """Read the content of a file from a given path; raise a ToolError if an error occurs."""
        try:
            # Double check that the file exists before attempting to read it
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            
            # Confirm it's a file and not a directory
            if not path.is_file():
                raise IsADirectoryError(f"Path is not a file: {path}")
                
            return path.read_text(encoding='utf-8', errors='replace')
        except Exception as e:
            raise ToolError(f"Ran into {e} while trying to read {path}") from None

    def write_file(self, path: Path, file: str):
        """Write the content of a file to a given path; raise a ToolError if an error occurs."""
        try:
            import os
            
            # Create parent directory if it doesn't exist
            if not path.parent.exists():
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                except Exception as mkdir_error:
                    raise ToolError(f"Failed to create directory {path.parent}: {mkdir_error}")
            
            # Double check parent directory exists before proceeding
            if not path.parent.exists():
                raise FileNotFoundError(f"Parent directory not found: {path.parent}")
                
            # Use utf-8 encoding with error handling
            # Filter out problematic characters for Windows systems if needed
            if os.name == 'nt':
                # Replace or remove any characters that might cause encoding issues on Windows
                import re
                # Pattern to match emojis and other problematic characters
                emoji_pattern = re.compile("["
                    u"\U0001F600-\U0001F64F"  # emoticons
                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                    u"\U0001F700-\U0001F77F"  # alchemical symbols
                    u"\U0001F780-\U0001F7FF"  # Geometric Shapes
                    u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                    u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                    u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                    u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                    u"\U00002702-\U000027B0"  # Dingbats
                    "]+", flags=re.UNICODE)
                file = emoji_pattern.sub(r'', file)
            
            # Write the file
            _ = path.write_text(file, encoding='utf-8')
            
            # Verify the file was written successfully
            if not path.exists():
                raise FileNotFoundError(f"File was not created: {path}")
                
            if not path.is_file():
                raise IsADirectoryError(f"Created path is not a file: {path}")
                
        except Exception as e:
            raise ToolError(f"Ran into {e} while trying to write to {path}") from None

    def _make_output(
        self,
        file_content: str,
        file_descriptor: str,
        init_line: int = 1,
        expand_tabs: bool = True,
    ):
        """Generate output for the CLI based on the content of a file."""
        file_content = maybe_truncate(file_content)
        if expand_tabs:
            file_content = file_content.expandtabs()
        file_content = "\n".join(
            [
                f"{i + init_line:6}\t{line}"
                for i, line in enumerate(file_content.split("\n"))
            ]
        )
        return (
            f"Here's the result of running `cat -n` on {file_descriptor}:\n"
            + file_content
            + "\n"
        )
