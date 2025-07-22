import asyncio
import sys
import os
from trae_agent.tools.edit_tool import TextEditorTool, ToolCallArguments

async def main():
    tool = TextEditorTool()
    current_dir = os.getcwd()
    
    print(f"Current directory: {current_dir}")
    
    # Test viewing the directory (using the actual current directory)
    print("\nTesting directory view...")
    result = await tool.execute({
        "command": "view", 
        "path": current_dir
    })
    print(result.output if hasattr(result, 'output') else result.error)
    
    # Test viewing a file
    print("\nTesting file view...")
    result = await tool.execute({
        "command": "view", 
        "path": os.path.join(current_dir, "test_file.txt")
    })
    print(result.output if hasattr(result, 'output') else result.error)
    
    # Test creating a file
    print("\nTesting file creation...")
    new_file_path = os.path.join(current_dir, "test_created_file2.txt")
    result = await tool.execute({
        "command": "create", 
        "path": new_file_path,
        "file_text": "This is a test created file"
    })
    print(result.output if hasattr(result, 'output') else result.error)
    
    # Test viewing a file with forward slashes (Git Bash style)
    print("\nTesting Git Bash style path...")
    # Convert Windows path to Git Bash style
    parts = current_dir.split(':')
    if len(parts) > 1:
        drive = parts[0].lower()
        rest = parts[1].replace('\\', '/')
        git_bash_path = f"/{drive}{rest}/test_file.txt"
        print(f"Converted path: {git_bash_path}")
        
        result = await tool.execute({
            "command": "view", 
            "path": git_bash_path
        })
        print(result.output if hasattr(result, 'output') else result.error)
        
    # Test creating a file with Git Bash style path
    print("\nTesting Git Bash style path creation...")
    if len(parts) > 1:
        drive = parts[0].lower()
        rest = parts[1].replace('\\', '/')
        git_bash_create_path = f"/{drive}{rest}/git_bash_test.txt"
        print(f"Git Bash create path: {git_bash_create_path}")
        
        result = await tool.execute({
            "command": "create", 
            "path": git_bash_create_path,
            "file_text": "This is a file created with Git Bash path"
        })
        print(result.output if hasattr(result, 'output') else result.error)
    
if __name__ == "__main__":
    asyncio.run(main())
