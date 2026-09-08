# Security Policy

## Supported Versions

We actively support the latest released version of `llm-reliability` with security updates and patches.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1.0 | :x:                |

## Reporting a Vulnerability

We take the security of `llm-reliability` and our users seriously. If you believe you have found a security vulnerability in this project, please report it to us as described below.

### Where to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities privately via:
- **GitHub Private Vulnerability Reporting**: Use the "Report a vulnerability" button on our GitHub repository's Security tab.
- **Email**: Contact the maintainers directly at `ashishuike8@gmail.com`.

### Information to Include

Please include as much of the following information as possible:
- Type of issue (e.g., buffer overflow, SQL injection, remote code execution, denial of service).
- Full paths of source file(s) related to the vulnerability.
- Step-by-step instructions to reproduce the issue.
- Proof-of-concept or exploit code, if available.
- Impact of the issue, including how an attacker could exploit it.

### Preferred Languages

We prefer communication in English.

### Policy & SLAs

- **Acknowledgement**: We will acknowledge receipt of your vulnerability report within **48 hours**.
- **Assessment**: We will confirm the vulnerability and determine its impact within **5 business days**.
- **Fix & Disclosure**: We will work on a fix and coordinate a public release along with a CVE identifier where appropriate. We request reasonable time to remediate before public disclosure.

## Security Practices

- **Zero Remote Dependencies by Default**: All analysis runs locally without transmitting prompts, completions, or traces to third-party endpoints.
- **Automated SAST**: Continuous Bandit static application security testing on every pull request and commit.
- **Dependency Scanning**: Weekly automated `pip-audit` CVE vulnerability scanning.
- **CodeQL**: Deep semantic code analysis with GitHub CodeQL.
- **Isolated Containers**: Official Docker images run with non-root privileges (`appuser:10001`).

