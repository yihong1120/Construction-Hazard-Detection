from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic import field_validator

from src.settings.base import EnvironmentSettings

DEFAULT_WEBSITE_HOSTS = {
    'main': 'changdar-server.mooo.com',
    'ai': 'ai.changdar-server.mooo.com',
    'wiki': 'wiki.changdar-server.mooo.com',
}


class WebsiteAccessSettings(EnvironmentSettings):
    status_file: Path = Field(
        Path('/run/website-access/status.json'),
        validation_alias='WEBSITE_ACCESS_STATUS_FILE',
    )
    hosts: dict[str, str] = Field(
        DEFAULT_WEBSITE_HOSTS,
        validation_alias='WEBSITE_ACCESS_HOSTS_JSON',
    )

    @field_validator('hosts')
    @classmethod
    def valid_hosts(cls, hosts):
        if hosts.keys() != DEFAULT_WEBSITE_HOSTS.keys():
            raise ValueError('Expected main, ai and wiki hostnames')
        for host in hosts.values():
            if len(host) > 253 or any(
                not part
                or len(part) > 63
                or part.startswith('-')
                or part.endswith('-')
                or not all(
                    c.isascii() and (c.isalnum() or c == '-') for c in part
                )
                for part in host.split('.')
            ):
                raise ValueError('Invalid website hostname')
        if len(set(hosts.values())) != 3:
            raise ValueError('Website hostnames must be distinct')
        return hosts


class InvitationSettings(EnvironmentSettings):
    policy_path: Path | None = Field(
        None,
        validation_alias='INVITATION_POLICY_PATH',
    )

    @field_validator('policy_path', mode='before')
    @classmethod
    def empty_path(cls, value):
        return None if value == '' else value
