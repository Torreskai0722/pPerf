"""Versioned model architecture profiles for semantic instrumentation."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Tuple

import jsonschema
import yaml


class ArchitectureProfileError(ValueError):
    """Raised when a profile is unknown or a model does not match it."""


@dataclass(frozen=True)
class MethodBinding:
    """One semantic method selected by an architecture profile."""

    owner: str
    target_path: str
    method: str
    required: bool

    @property
    def method_path(self) -> str:
        """Return the complete, stable path used in trace evidence."""
        prefix = self.target_path or "model"
        return f"{prefix}.{self.method}"

    def definition(self) -> Dict[str, object]:
        """Return the canonical serializable binding definition."""
        return {
            "method": self.method,
            "owner": self.owner,
            "required": self.required,
            "target_path": self.target_path,
        }


@dataclass(frozen=True)
class ModuleBinding:
    """One semantic module selected by an architecture profile."""

    module_path: str
    required: bool
    entrypoint: str

    def definition(self) -> Dict[str, object]:
        """Return the canonical serializable binding definition."""
        return {
            "entrypoint": self.entrypoint,
            "module_path": self.module_path,
            "required": self.required,
        }


@dataclass(frozen=True)
class ArchitectureProfile:
    """Immutable accepted structure and semantic instrumentation bindings."""

    name: str
    accepted_mro: Tuple[str, ...]
    method_bindings: Tuple[MethodBinding, ...]
    module_bindings: Tuple[ModuleBinding, ...]

    def definition(self) -> Dict[str, object]:
        """Return the complete definition used to derive the profile hash."""
        return {
            "accepted_mro": list(self.accepted_mro),
            "method_bindings": [
                binding.definition() for binding in self.method_bindings
            ],
            "module_bindings": [
                binding.definition() for binding in self.module_bindings
            ],
            "name": self.name,
        }

    @property
    def sha256(self) -> str:
        """Return SHA-256 of the canonical complete profile definition."""
        canonical = json.dumps(
            self.definition(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def profile_schema_path() -> Path:
    """Return the bundled architecture-profile schema."""
    return (
        Path(__file__).with_name("schema")
        / "architecture_profile.schema.json"
    )


def default_profile_directory() -> Path:
    """Return the bundled architecture-profile configuration directory."""
    return Path(__file__).with_name("config") / "architecture_profiles"


def _binding_duplicates(values, key):
    """Return sorted duplicate binding identities."""
    seen = set()
    duplicates = set()
    for value in values:
        identity = key(value)
        if identity in seen:
            duplicates.add(identity)
        seen.add(identity)
    return sorted(duplicates)


def _parse_profile(
        path: Path, schema: Dict[str, object]
) -> ArchitectureProfile:
    """Load and validate one architecture-profile YAML file."""
    try:
        parsed = yaml.safe_load(path.read_bytes())
    except (OSError, yaml.YAMLError) as exc:
        raise ArchitectureProfileError(
            f"cannot read architecture profile {path}: {exc}"
        ) from exc
    try:
        jsonschema.Draft202012Validator(schema).validate(parsed)
    except jsonschema.ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path)
        prefix = f"{location}: " if location else ""
        raise ArchitectureProfileError(
            f"invalid architecture profile {path.name}: "
            f"{prefix}{exc.message}"
        ) from exc
    if path.stem != parsed["name"]:
        raise ArchitectureProfileError(
            f"architecture profile filename {path.stem!r} does not match "
            f"name {parsed['name']!r}"
        )
    method_duplicates = _binding_duplicates(
        parsed["method_bindings"],
        lambda value: (value["target_path"], value["method"]),
    )
    if method_duplicates:
        raise ArchitectureProfileError(
            f"architecture profile {parsed['name']!r} has duplicate method "
            f"bindings: {method_duplicates}"
        )
    module_duplicates = _binding_duplicates(
        parsed["module_bindings"], lambda value: value["module_path"]
    )
    if module_duplicates:
        raise ArchitectureProfileError(
            f"architecture profile {parsed['name']!r} has duplicate module "
            f"bindings: {module_duplicates}"
        )
    return ArchitectureProfile(
        name=parsed["name"],
        accepted_mro=tuple(parsed["accepted_mro"]),
        method_bindings=tuple(
            MethodBinding(
                binding["owner"], binding["target_path"], binding["method"],
                binding["required"]
            )
            for binding in parsed["method_bindings"]
        ),
        module_bindings=tuple(
            ModuleBinding(
                binding["module_path"], binding["required"],
                binding["entrypoint"]
            )
            for binding in parsed["module_bindings"]
        ),
    )


def load_architecture_profiles(
        directory: Optional[Path] = None
) -> Mapping[str, ArchitectureProfile]:
    """Load the immutable architecture-profile registry from YAML."""
    root = (
        Path(directory)
        if directory is not None
        else default_profile_directory()
    )
    try:
        schema = json.loads(profile_schema_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArchitectureProfileError(
            f"cannot read architecture profile schema: {exc}"
        ) from exc
    paths = sorted(root.glob("*.yaml"))
    if not paths:
        raise ArchitectureProfileError(
            f"no architecture profiles found in {root}"
        )
    profiles = {}
    for path in paths:
        profile = _parse_profile(path, schema)
        if profile.name in profiles:
            raise ArchitectureProfileError(
                f"duplicate architecture profile name {profile.name!r}"
            )
        profiles[profile.name] = profile
    return MappingProxyType(profiles)


ARCHITECTURE_PROFILES = load_architecture_profiles()


def get_architecture_profile(name: str) -> ArchitectureProfile:
    """Resolve a named architecture profile or reject it clearly."""
    try:
        return ARCHITECTURE_PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(ARCHITECTURE_PROFILES))
        raise ArchitectureProfileError(
            f"unknown architecture profile {name!r}; "
            f"expected one of: {choices}"
        ) from exc


def architecture_profile_metadata(name: str) -> Dict[str, object]:
    """Return the status/manifest metadata for a configured profile."""
    profile = get_architecture_profile(name)
    return {
        "architecture_profile": profile.name,
        "architecture_profile_sha256": profile.sha256,
        "required_method_bindings": sorted(
            binding.method_path for binding in profile.method_bindings
            if binding.required
        ),
        "required_module_bindings": sorted(
            binding.module_path for binding in profile.module_bindings
            if binding.required
        ),
        "observed_method_bindings": [],
        "observed_module_bindings": [],
    }
