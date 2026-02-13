"""
Log reading tools for the AI agent
"""
import os
import re
from pathlib import Path
from langchain_core.tools import tool
from ..config import Config


_DEFAULT_MAX_BYTES = int(os.getenv("MAX_LOG_BYTES", "200000"))
_DEFAULT_MAX_LINES = int(os.getenv("MAX_LOG_LINES", "2000"))


def _redact_secrets(text: str) -> str:
    # Keep this lightweight; can be expanded later.
    patterns = [
        # OpenAI/OpenRouter style keys
        r"sk-or-v1-[A-Za-z0-9\-_]{10,}",
        r"sk-[A-Za-z0-9\-_]{10,}",
        # Generic API key assignments
        r"(?i)(api[_-]?key\s*[:=]\s*)([^\s\"']{8,})",
        r"(?i)(authorization\s*[:=]\s*bearer\s+)([^\s\"']{8,})",
        # AWS access key id (best-effort)
        r"AKIA[0-9A-Z]{16}",
    ]

    redacted = text
    redacted = re.sub(patterns[0], "[REDACTED_OPENROUTER_KEY]", redacted)
    redacted = re.sub(patterns[1], "[REDACTED_API_KEY]", redacted)
    redacted = re.sub(patterns[2], r"\1[REDACTED]", redacted)
    redacted = re.sub(patterns[3], r"\1[REDACTED]", redacted)
    redacted = re.sub(patterns[4], "[REDACTED_AWS_ACCESS_KEY_ID]", redacted)
    return redacted


def _resolve_log_path(filename: str) -> Path | None:
    logs_dir = Path(Config.LOG_DIRECTORY).resolve()

    # Disallow absolute paths and path traversal.
    candidate = (logs_dir / filename).resolve()
    if logs_dir != candidate and logs_dir not in candidate.parents:
        return None
    return candidate


def _read_capped_text(path: Path, max_bytes: int, max_lines: int) -> tuple[str, bool, bool]:
    truncated_bytes = False
    truncated_lines = False

    raw = path.read_text(encoding="utf-8", errors="replace")
    if len(raw.encode("utf-8", errors="replace")) > max_bytes:
        raw = raw.encode("utf-8", errors="replace")[:max_bytes].decode("utf-8", errors="replace")
        truncated_bytes = True

    lines = raw.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated_lines = True

    return "\n".join(lines), truncated_bytes, truncated_lines


@tool
def read_log_file(filename: str, max_bytes: int = _DEFAULT_MAX_BYTES, max_lines: int = _DEFAULT_MAX_LINES, redact: bool = True) -> str:
    """
    Read contents of a log file from the logs directory.
    
    Args:
        filename: Name of the log file (e.g., 'app.log', 'k8s.log', 'error.log')
    
    Returns:
        String containing the log file contents, or error message if file not found
    """
    log_path = _resolve_log_path(filename)
    if log_path is None:
        return "Error: Invalid filename (path traversal is not allowed)"
    
    try:
        if not log_path.exists() or not log_path.is_file():
            raise FileNotFoundError

        content, truncated_bytes, truncated_lines = _read_capped_text(log_path, max_bytes=max_bytes, max_lines=max_lines)
        if redact:
            content = _redact_secrets(content)
        
        # Add metadata
        file_size = os.path.getsize(log_path)
        line_count = content.count('\n') + 1
        
        header = f"File: {filename}\nSize: {file_size} bytes\nLines: {line_count}"
        notes = []
        if truncated_bytes:
            notes.append(f"truncated to {max_bytes} bytes")
        if truncated_lines:
            notes.append(f"truncated to first {max_lines} lines")
        if notes:
            header += "\nNote: " + ", ".join(notes)

        return f"{header}\n\n{content}"
    
    except FileNotFoundError:
        return f"Error: Log file '{filename}' not found in {Config.LOG_DIRECTORY}/ directory"
    except PermissionError:
        return f"Error: Permission denied reading '{filename}'"
    except Exception as e:
        return f"Error reading '{filename}': {str(e)}"


@tool
def list_log_files() -> str:
    """
    List all available log files in the logs directory.
    
    Returns:
        String containing list of available log files with their sizes
    """
    log_dir = Path(Config.LOG_DIRECTORY)
    
    if not log_dir.exists():
        return f"Error: Log directory '{Config.LOG_DIRECTORY}' does not exist"
    
    try:
        log_files = [f for f in log_dir.iterdir() if f.is_file() and f.suffix == '.log']
        
        if not log_files:
            return f"No .log files found in {Config.LOG_DIRECTORY}/ directory"
        
        result = f"Available log files in {Config.LOG_DIRECTORY}/:\n\n"
        for log_file in sorted(log_files):
            size = log_file.stat().st_size
            size_kb = size / 1024
            result += f"  - {log_file.name} ({size_kb:.2f} KB)\n"
        
        return result
    
    except Exception as e:
        return f"Error listing log files: {str(e)}"


@tool
def search_logs(filename: str, search_term: str) -> str:
    """
    Search for a specific term in a log file and return matching lines.
    
    Args:
        filename: Name of the log file to search
        search_term: Term to search for (case-insensitive)
    
    Returns:
        String containing matching log lines with line numbers
    """
    log_path = _resolve_log_path(filename)
    if log_path is None:
        return "Error: Invalid filename (path traversal is not allowed)"
    
    try:
        if not log_path.exists() or not log_path.is_file():
            raise FileNotFoundError

        # Read with a cap to avoid huge memory usage
        content, _, _ = _read_capped_text(log_path, max_bytes=_DEFAULT_MAX_BYTES, max_lines=_DEFAULT_MAX_LINES)
        lines = content.splitlines()
        
        matches = []
        for line_num, line in enumerate(lines, 1):
            if search_term.lower() in line.lower():
                matches.append(f"Line {line_num}: {line.rstrip()}")
        
        if not matches:
            return f"No matches found for '{search_term}' in {filename}"
        
        result = f"Found {len(matches)} matches for '{search_term}' in {filename}:\n\n"
        result += '\n'.join(matches)
        
        return result
    
    except FileNotFoundError:
        return f"Error: Log file '{filename}' not found"
    except Exception as e:
        return f"Error searching '{filename}': {str(e)}"


def get_log_tools() -> list:
    """
    Get all log-related tools for the agent.
    
    Returns:
        List of tool functions
    """
    return [read_log_file, list_log_files, search_logs]