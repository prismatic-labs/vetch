# Security Pre-Commit Hooks

This document explains the security scanning hooks configured for Vetch.

---

## Setup

### 1. Install pre-commit

```bash
pip install pre-commit
```

### 2. Install the hooks

```bash
pre-commit install
```

This will automatically run security checks before every commit.

---

## Configured Hooks

### 🔧 Ruff (Linting & Formatting)
- **Purpose:** Code quality and style consistency
- **Auto-fix:** Yes
- **Config:** `.ruff.toml` or `pyproject.toml`

### 🔍 MyPy (Type Checking)
- **Purpose:** Static type checking with `--strict` mode
- **Auto-fix:** No
- **Config:** `pyproject.toml`

### 🛡️ Bandit (Security Scanning)
- **Purpose:** Find common security issues in Python code
- **Auto-fix:** No
- **Config:** `pyproject.toml` - `[tool.bandit]`

**Skipped Checks (with justification):**
- `B101` - Assert statements (used for mypy narrowing, not production logic)
- `B110/B112` - Try-except-pass (intentional fail-open architecture)
- `B311` - random.random() (legitimate use for jitter, not cryptographic)
- `B310` - urllib.urlopen (validated HTTPS URLs only)

### 🔑 Detect-Secrets (Secret Scanning)
- **Purpose:** Prevent committing API keys, passwords, tokens
- **Auto-fix:** No
- **Baseline:** `.secrets.baseline`

---

## Running Manually

### Run all hooks on all files
```bash
pre-commit run --all-files
```

### Run specific hook
```bash
pre-commit run bandit --all-files
pre-commit run detect-secrets --all-files
```

### Run only on staged files (default)
```bash
pre-commit run
```

---

## Common Workflows

### Adding a new secret (intentionally)

If you need to add a test fixture with a fake API key:

1. Add the secret to the file
2. Run: `detect-secrets scan --baseline .secrets.baseline`
3. Review and commit both the file and updated baseline

### Fixing a Bandit warning

If Bandit flags a legitimate issue:

1. **Option A:** Fix the code (preferred)
   ```python
   # Before (flagged by Bandit)
   password = "hardcoded123"  # B105: hardcoded password

   # After (secure)
   password = os.environ.get("DB_PASSWORD")
   ```

2. **Option B:** Add inline comment if false positive
   ```python
   # Not a real password - test fixture
   test_password = "fake-password-123"  # nosec B105
   ```

3. **Option C:** Add to skip list in `pyproject.toml` (requires justification)

### Bypassing hooks (emergency only)

```bash
git commit --no-verify -m "Emergency hotfix"
```

⚠️ **WARNING:** Only use `--no-verify` in genuine emergencies. All commits should pass security checks before merging.

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Security Scan

on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install bandit[toml] detect-secrets

      - name: Run Bandit
        run: bandit -c pyproject.toml -r src/

      - name: Run Detect-Secrets
        run: detect-secrets scan --baseline .secrets.baseline
```

---

## Updating Hooks

Pre-commit hooks are versioned. To update to latest versions:

```bash
pre-commit autoupdate
```

This will update the `rev` fields in `.pre-commit-config.yaml`.

---

## Troubleshooting

### Hook fails with "command not found"

**Solution:** Reinstall hooks
```bash
pre-commit clean
pre-commit install
```

### Bandit flags too many false positives

**Solution:** Review and update `[tool.bandit]` skips in `pyproject.toml`

```toml
[tool.bandit]
exclude_dirs = ["tests", "docs"]
skips = ["B101", "B110"]  # Add codes to skip with justification
```

### Detect-secrets flags legitimate code

**Solution:** Update baseline
```bash
detect-secrets scan --baseline .secrets.baseline
git add .secrets.baseline
git commit -m "chore: update secrets baseline"
```

### Pre-commit is too slow

**Solution:** Run only on changed files (default) or skip in development
```bash
# Skip hooks during commit
git commit --no-verify

# Run hooks in CI only (not recommended)
# Remove: pre-commit install
```

---

## Best Practices

### ✅ DO

- Run `pre-commit run --all-files` before pushing
- Keep hooks updated with `pre-commit autoupdate`
- Review Bandit warnings carefully - they often catch real issues
- Add justification comments when using `# nosec`
- Update `.secrets.baseline` when adding test fixtures

### ❌ DON'T

- Use `--no-verify` routinely (only for emergencies)
- Blindly skip Bandit warnings without understanding them
- Commit secrets (even test secrets that look real)
- Disable hooks permanently in CI

---

## Security Contacts

If you discover a security vulnerability:

1. **DO NOT** open a public GitHub issue
2. Email: security@prismatic-labs.com (if applicable)
3. Or: Use GitHub Security Advisories (private reporting)

---

## Additional Resources

- [Bandit Documentation](https://bandit.readthedocs.io/)
- [Detect-Secrets Documentation](https://github.com/Yelp/detect-secrets)
- [Pre-commit Documentation](https://pre-commit.com/)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)

---

## Maintenance

### Monthly Tasks

- [ ] Run `pre-commit autoupdate` to get latest hook versions
- [ ] Review and update `[tool.bandit]` skips if needed
- [ ] Audit `.secrets.baseline` for stale entries

### Before Major Releases

- [ ] Run full security scan: `bandit -r src/ -f json -o security_report.json`
- [ ] Review all Bandit warnings (including skipped ones)
- [ ] Update `SECURITY_AUDIT.md` with findings
- [ ] Consider external security review for v1.0+
