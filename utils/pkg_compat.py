"""Compatibility for environments where setuptools no longer includes pkg_resources.

Only the subset referenced in the inherited YOLOv9 utilities is implemented.
"""
from importlib import metadata
from packaging.requirements import Requirement
from packaging.version import parse as parse_version


class VersionConflict(Exception):
    pass


class DistributionNotFound(Exception):
    pass


def parse_requirements(lines):
    for line in lines:
        line = line.partition('#')[0].strip()
        if line and (not line.startswith(('-', 'git+'))):
            yield Requirement(line)


def require(spec):
    req = Requirement(spec)
    try:
        installed = metadata.version(req.name)
    except metadata.PackageNotFoundError as exc:
        raise DistributionNotFound(str(req)) from exc
    if req.specifier and (not req.specifier.contains(installed)):
        raise VersionConflict(f'{req.name} {installed} does not meet {req.specifier}')
