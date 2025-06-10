from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.utils.logging import logger


@dataclass
class PromptMetadata:
    """Metadata for a prompt template"""

    name: str
    version: str
    description: str
    author: str
    created_date: str
    last_modified: str
    tags: List[str]


@dataclass
class PromptConfig:
    """Configuration for a prompt"""

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    model_type: Optional[str] = None


@dataclass
class PromptVariable:
    """Variable definition for a prompt"""

    name: str
    type: str
    required: bool
    description: str


@dataclass
class PromptTemplate:
    """Complete prompt template with metadata and configuration"""

    metadata: PromptMetadata
    config: PromptConfig
    variables: List[PromptVariable]
    prompt_template: str

    def render(self, **kwargs) -> str:
        """Render the prompt template with provided variables"""
        # Validate required variables
        required_vars = {var.name for var in self.variables if var.required}
        provided_vars = set(kwargs.keys())
        missing_vars = required_vars - provided_vars

        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")

        # Simple template rendering using string formatting
        template = self.prompt_template
        for key, value in kwargs.items():
            template = template.replace(f"{{{{{key}}}}}", str(value))

        return template


class PromptManager:
    """Prompt management system"""

    def __init__(self, prompts_dir: Optional[str] = None):
        self.prompts_dir = Path(prompts_dir or "prompts")
        self._prompt_cache: Dict[str, PromptTemplate] = {}
        self._load_all_prompts()

    def _load_all_prompts(self):
        """Load all prompt templates from the prompts directory"""
        if not self.prompts_dir.exists():
            logger.warning(f"Prompts directory not found: {self.prompts_dir}")
            return

        # Find all YAML files recursively
        yaml_files = list(self.prompts_dir.rglob("*.yaml")) + list(
            self.prompts_dir.rglob("*.yml")
        )

        for yaml_file in yaml_files:
            try:
                self._load_prompt_file(yaml_file)
            except Exception as e:
                logger.error(f"Failed to load prompt file {yaml_file}: {e}")

    def _load_prompt_file(self, file_path: Path):
        """Load a single prompt file"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Parse metadata
        metadata_data = data.get("metadata", {})
        metadata = PromptMetadata(
            name=metadata_data.get("name", file_path.stem),
            version=metadata_data.get("version", "1.0.0"),
            description=metadata_data.get("description", ""),
            author=metadata_data.get("author", ""),
            created_date=metadata_data.get("created_date", ""),
            last_modified=metadata_data.get("last_modified", ""),
            tags=metadata_data.get("tags", []),
        )

        # Parse config
        config_data = data.get("config", {})
        config = PromptConfig(
            temperature=config_data.get("temperature"),
            max_tokens=config_data.get("max_tokens"),
            model_type=config_data.get("model_type"),
        )

        # Parse variables
        variables_data = data.get("variables", [])
        variables = [
            PromptVariable(
                name=var.get("name", ""),
                type=var.get("type", "string"),
                required=var.get("required", False),
                description=var.get("description", ""),
            )
            for var in variables_data
        ]
        # Create prompt template
        prompt_template = PromptTemplate(
            metadata=metadata,
            config=config,
            variables=variables,
            prompt_template=data.get("prompt_template", ""),
        )
        # Cache the prompt
        self._prompt_cache[metadata.name] = prompt_template
        logger.debug(f"Loaded prompt template: {metadata.name}")

    def get_prompt(self, name: str) -> Optional[PromptTemplate]:
        """Get a prompt template by name"""
        return self._prompt_cache.get(name)

    def render_prompt(self, name: str, **kwargs) -> str:
        """Render a prompt template with variables"""
        prompt_template = self.get_prompt(name)
        if not prompt_template:
            raise ValueError(f"Prompt template '{name}' not found")

        return prompt_template.render(**kwargs)

    def list_prompts(self) -> List[str]:
        """List all available prompt names"""
        return list(self._prompt_cache.keys())

    def get_prompt_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a prompt template"""
        prompt_template = self.get_prompt(name)
        if not prompt_template:
            return None

        return {
            "metadata": {
                "name": prompt_template.metadata.name,
                "version": prompt_template.metadata.version,
                "description": prompt_template.metadata.description,
                "author": prompt_template.metadata.author,
                "created_date": prompt_template.metadata.created_date,
                "last_modified": prompt_template.metadata.last_modified,
                "tags": prompt_template.metadata.tags,
            },
            "config": {
                "temperature": prompt_template.config.temperature,
                "max_tokens": prompt_template.config.max_tokens,
                "model_type": prompt_template.config.model_type,
            },
            "variables": [
                {
                    "name": var.name,
                    "type": var.type,
                    "required": var.required,
                    "description": var.description,
                }
                for var in prompt_template.variables
            ],
        }

    def reload_prompts(self):
        """Reload all prompt templates"""
        self._prompt_cache.clear()
        self._load_all_prompts()
        logger.info("Reloaded all prompt templates")


# Global prompt manager instance
prompt_manager = PromptManager()
