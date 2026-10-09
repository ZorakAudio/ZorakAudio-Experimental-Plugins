"""One release policy shared by build selection, shard validation and packaging."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLATFORMS = ('windows', 'macos', 'linux')
COLLECTIONS = ('JoepVanlier', 'Essentials', 'All')


def load_policy(repo_root=ROOT):
    path = repo_root / 'release-collections.json'
    policy = json.loads(path.read_text(encoding='utf-8'))
    if policy['schemaVersion'] != 1:
        raise ValueError('Unsupported release collection policy version')
    return policy


def policy_digest(repo_root=ROOT):
    """Hash the policy identically after LF or Windows CRLF checkout."""
    data = (repo_root / 'release-collections.json').read_bytes().replace(b'\r\n', b'\n')
    return hashlib.sha256(data).hexdigest()


def policy_receipt_digests(repo_root=ROOT):
    """Accept earlier raw-byte receipts for exactly this policy's LF/CRLF forms.

    This recovers already-built Windows shards without accepting a different
    policy or disabling the gate. New receipts always use policy_digest().
    """
    data = (repo_root / 'release-collections.json').read_bytes().replace(b'\r\n', b'\n')
    return {hashlib.sha256(data).hexdigest(),
            hashlib.sha256(data.replace(b'\n', b'\r\n')).hexdigest()}


def build_catalog(specs, repo_root=ROOT):
    policy = load_policy(repo_root)
    by_slug = {spec.slug: spec for spec in specs}
    excluded = set(policy['buildExcluded'])
    essential = [item['slug'] for item in policy['essentials']]
    reviewed = set(policy['essentialsExcluded'])
    unknown = (excluded | set(essential) | reviewed) - by_slug.keys()
    if unknown:
        raise ValueError('Unknown plugin in release policy: ' + ', '.join(sorted(unknown)))
    if len(essential) != len(set(essential)) or set(essential) & (excluded | reviewed):
        raise ValueError('Conflicting Essentials selection in release policy')
    if any(by_slug[slug].category == 'JoepVanlier' for slug in essential):
        raise ValueError('JoepVanlier plugins cannot be in Essentials')
    return [spec for spec in specs if spec.slug not in excluded]


def collections_for(specs, repo_root=ROOT):
    eligible = build_catalog(specs, repo_root)
    by_slug = {spec.slug: spec for spec in eligible}
    policy = load_policy(repo_root)
    return {
        'JoepVanlier': [spec for spec in eligible if spec.category == 'JoepVanlier'],
        'Essentials': [by_slug[item['slug']] for item in policy['essentials']],
        'All': [spec for spec in eligible if spec.category != 'JoepVanlier'],
    }
