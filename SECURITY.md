# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to: marco@prismaticlabs.ai

You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

Please include the following information:
- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

## Security Design Principles

### Privacy First
- Vetch NEVER reads prompt or completion content
- Only metadata is captured: model name, token counts, region, latency
- No user identifiers are tracked or hashed

### Fail-Open
- If Vetch fails, the LLM call always proceeds
- Errors are logged but never block the host application
- Kill switch available: `VETCH_DISABLED=true`

### Data Storage
- Local SQLite database stored in `~/.vetch/` (user's home directory)
- No data is transmitted externally unless explicitly configured
- HTTP emission requires explicit opt-in: `VETCH_ENABLE_REMOTE=true`

### Dependencies
- Zero runtime dependencies (stdlib only)
- Minimizes supply chain attack surface

## Known Limitations (Alpha)

1. **Energy estimates are uncertain**: Tier 3 estimates have order of magnitude uncertainty
2. **Region inference**: Timezone-based inference is ~30% accurate
3. **No encryption**: Local SQLite database is not encrypted at rest

## Security Checklist for Contributors

- [ ] No hardcoded credentials or API keys
- [ ] All user input is validated
- [ ] SQL queries use parameterized statements
- [ ] File paths are validated against traversal attacks
- [ ] HTTP requests have timeouts configured
- [ ] Sensitive data is not logged
