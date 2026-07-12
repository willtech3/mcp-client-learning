# mcp-client-learning task runner

default:
    @just --list

# Install dependencies
install:
    uv sync --dev

# Run the test suite
test:
    uv run pytest

# Lint (includes bandit security rules)
lint:
    uv run ruff check mcp_client tests

# Auto-fix lint issues
lint-fix:
    uv run ruff check --fix mcp_client tests

# Format code
format:
    uv run ruff format mcp_client tests

# Type-check
typecheck:
    uv run pyright

# All quality gates
check: lint typecheck test
    @echo "All checks passed"

# Guided protocol demo against a local Virtual Library server
demo-local:
    uv run mcp-client demo --server http://127.0.0.1:8080/mcp --anonymous

# Demo against a 2026-07-28 (modern era) server; compat probes with server/discover
demo-modern:
    uv run mcp-client demo --server http://127.0.0.1:8080/mcp --anonymous

# Pretty-print a modern server's server/discover capability document
discover:
    uv run mcp-client discover --server http://127.0.0.1:8080/mcp --anonymous
