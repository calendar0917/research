"""rr — research remote execution interface.

Layered on top of:
  Agent -> Skill -> rr -> backend abstraction -> SSH / rsync / Git / Slurm / uv

Design goals (in priority order): usability, stability/recoverability, research
provenance, maintainability, extensibility.  No daemon, no database, no
interactive shell dependency.  Authoritative state lives on the remote
filesystem (meta.json) and in Slurm / Git.
"""

__version__ = "0.2.1"
SCHEMA = "rr/2"
