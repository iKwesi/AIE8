"""Configuration utilities for API keys and environment variables."""

import os
import getpass
from typing import Optional
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()


def get_api_key(
    key_name: str,
    prompt_message: Optional[str] = None,
    required: bool = True
) -> Optional[str]:
    """Get API key from environment or prompt user.
    
    This function first attempts to load the API key from environment variables
    (which can be set via a .env file or system environment). If not found,
    it falls back to prompting the user for input.
    
    Args:
        key_name: Environment variable name (e.g., 'OPENAI_API_KEY')
        prompt_message: Custom prompt message for getpass. If None, a default
            message will be generated based on the key_name.
        required: Whether the key is required. If True and no key is provided,
            raises ValueError. If False, returns None when no key is provided.
        
    Returns:
        API key string or None if optional and not provided
        
    Raises:
        ValueError: If required=True and no key is provided
        
    Examples:
        >>> # Get required API key (will prompt if not in environment)
        >>> api_key = get_api_key("OPENAI_API_KEY", required=True)
        
        >>> # Get optional API key (returns None if not provided)
        >>> tavily_key = get_api_key("TAVILY_API_KEY", required=False)
        
        >>> # Custom prompt message
        >>> key = get_api_key(
        ...     "CUSTOM_API_KEY",
        ...     prompt_message="Enter your custom API key: "
        ... )
    """
    # Try to get from environment first
    api_key = os.getenv(key_name)
    
    if api_key:
        print(f"✓ {key_name} loaded from environment")
        return api_key
    
    # Fall back to prompting user
    if prompt_message is None:
        if required:
            prompt_message = f"{key_name}: "
        else:
            prompt_message = f"{key_name} (optional - press Enter to skip): "
    
    try:
        api_key = getpass.getpass(prompt_message)
        if api_key.strip():
            # Set in environment for this session
            os.environ[key_name] = api_key
            print(f"✓ {key_name} set from user input")
            return api_key
        elif required:
            raise ValueError(f"{key_name} is required but not provided")
        else:
            print(f"⚠ Skipping optional {key_name}")
            return None
    except Exception as e:
        if required:
            raise ValueError(f"Failed to get {key_name}: {e}")
        print(f"⚠ Skipping optional {key_name}")
        return None


def setup_api_keys(
    require_openai: bool = True,
    require_tavily: bool = False,
    require_langsmith: bool = False,
    require_guardrails: bool = False
) -> dict:
    """Set up all API keys with sensible defaults.
    
    This is a convenience function that sets up all common API keys
    used in the project. It returns a dictionary of the keys that
    were successfully configured.
    
    Args:
        require_openai: Whether OpenAI API key is required (default: True)
        require_tavily: Whether Tavily API key is required (default: False)
        require_langsmith: Whether LangSmith API key is required (default: False)
        require_guardrails: Whether Guardrails API key is required (default: False)
        
    Returns:
        Dictionary mapping key names to their values (or None if not set)
        
    Example:
        >>> keys = setup_api_keys(require_openai=True, require_tavily=False)
        >>> if keys['LANGCHAIN_API_KEY']:
        ...     os.environ["LANGCHAIN_TRACING_V2"] = "true"
    """
    keys = {}
    
    # OpenAI API Key
    keys['OPENAI_API_KEY'] = get_api_key(
        "OPENAI_API_KEY",
        prompt_message="OpenAI API Key: ",
        required=require_openai
    )
    
    # Tavily API Key (for web search)
    keys['TAVILY_API_KEY'] = get_api_key(
        "TAVILY_API_KEY",
        prompt_message="Tavily API Key (optional - press Enter to skip): ",
        required=require_tavily
    )
    
    # LangSmith API Key (for tracing)
    keys['LANGCHAIN_API_KEY'] = get_api_key(
        "LANGCHAIN_API_KEY",
        prompt_message="LangChain API Key (optional - press Enter to skip): ",
        required=require_langsmith
    )
    
    # Guardrails API Key
    keys['GUARDRAILS_API_KEY'] = get_api_key(
        "GUARDRAILS_API_KEY",
        prompt_message="Guardrails API Key (optional - press Enter to skip): ",
        required=require_guardrails
    )
    
    return keys
