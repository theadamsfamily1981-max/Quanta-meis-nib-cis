# CLAUDE.md - AI Assistant Guide for Quanta-meis-nib-cis

Last Updated: 2025-11-14

## Repository Overview

**Repository Name:** Quanta-meis-nib-cis
**Purpose:** Research for quanta meis nib cis code
**Status:** Early-stage/initialization phase
**Owner:** theadamsfamily1981-max

This repository is currently in its early stages with minimal content. It serves as a research repository for quantum-related code development.

## Current Repository State

### Structure
```
Quanta-meis-nib-cis/
├── .git/                    # Git version control
├── CLAUDE.md                # This file - AI assistant guide
└── README.md                # Project documentation
```

### Repository Statistics
- **Total Size:** ~151 KB (mostly git metadata)
- **Source Files:** 0 (repository is awaiting code implementation)
- **Documentation:** 2 files (README.md, CLAUDE.md)
- **Initialized:** October 30, 2025

### Technology Stack
**Current:** None (repository is empty)
**Expected:** To be determined based on quantum computing research requirements

## Git Configuration & Workflow

### Remote Configuration
- **Remote URL:** `http://local_proxy@127.0.0.1:33516/git/theadamsfamily1981-max/Quanta-meis-nib-cis`
- **GPG Signing:** Enabled (SSH format)
- **Git User:** Claude (noreply@anthropic.com)

### Branch Strategy

**Current Branch:** `claude/create-codebase-documentation-01PN8mRM5v7awgD1pSc4hEJe`

**Workflow Notes:**
- The README indicates: "commit directly to main"
- This suggests a simplified workflow for direct commits
- Claude Code tasks use branches with pattern: `claude/<task-description>-<session-id>`
- All Claude branches must start with 'claude/' and end with matching session ID

### Commit History
```
f4fd90b - 2025-10-30 23:09:24 - "Update README with commit instructions"
29882d4 - 2025-10-30 22:54:58 - "Initial commit"
```

## Development Guidelines for AI Assistants

### General Principles

1. **Early-Stage Awareness**
   - This repository is currently empty of source code
   - When adding code, establish clear project structure from the start
   - Consider best practices for the chosen technology stack
   - Set up proper dependency management early

2. **Code Quality Standards**
   - Follow language-specific best practices when code is added
   - Implement proper error handling
   - Write self-documenting code with clear variable/function names
   - Add comments for complex logic or quantum-specific algorithms

3. **Security Considerations**
   - Avoid security vulnerabilities (XSS, SQL injection, command injection, etc.)
   - Never commit secrets, API keys, or credentials
   - Use environment variables for sensitive configuration
   - Review OWASP Top 10 when implementing web-facing features

### File Operations Best Practices

1. **Reading Files**
   - Always use the Read tool for file contents (not cat/head/tail)
   - Check file existence before operations when necessary

2. **Editing Files**
   - Prefer editing existing files over creating new ones
   - Use the Edit tool for modifications (not sed/awk)
   - Preserve exact indentation and formatting

3. **Creating Files**
   - Only create new files when absolutely necessary
   - Use the Write tool (not echo or heredoc redirects)
   - Follow consistent naming conventions

4. **Searching**
   - Use Grep for content searches (not grep/rg commands)
   - Use Glob for file pattern matching (not find/ls)
   - Use Task tool with Explore subagent for complex codebase exploration

### Git Operations

#### Committing Changes

**When to Commit:**
- Only create commits when explicitly requested by the user
- Never commit changes proactively without user approval
- If unclear, ask the user first

**Commit Process:**
1. Run `git status` to see untracked files
2. Run `git diff` to see staged and unstaged changes
3. Review recent `git log` to match commit message style
4. Draft concise commit message (1-2 sentences, focus on "why" not "what")
5. Stage relevant files
6. Create commit
7. Verify with `git status`

**Commit Message Format:**
- Use imperative mood ("Add feature" not "Added feature")
- Be concise but descriptive
- Focus on the purpose of changes
- Follow existing commit message patterns in the repository

#### Pushing Changes

**Push Command:**
```bash
git push -u origin <branch-name>
```

**Critical Requirements:**
- Branch must start with 'claude/' and end with matching session ID
- Retry up to 4 times with exponential backoff (2s, 4s, 8s, 16s) on network errors
- Only retry on network errors, not auth failures

**Example:**
```bash
git push -u origin claude/create-codebase-documentation-01PN8mRM5v7awgD1pSc4hEJe
```

#### Fetching/Pulling

**Fetch Specific Branch:**
```bash
git fetch origin <branch-name>
```

**Pull Changes:**
```bash
git pull origin <branch-name>
```

**Retry Logic:**
- Retry up to 4 times with exponential backoff on network failures

### Task Management

**Use TodoWrite Tool:**
- Create todo lists for multi-step tasks (3+ steps)
- Mark tasks as `in_progress` before starting work
- Mark tasks as `completed` immediately after finishing
- Only one task should be `in_progress` at a time
- Update todos in real-time as work progresses

**Task States:**
- `pending`: Not yet started
- `in_progress`: Currently working on (only ONE at a time)
- `completed`: Task finished successfully

**When NOT to Use Todos:**
- Single, straightforward tasks
- Trivial operations
- Purely conversational requests

## Future Development Recommendations

### When Adding Code

1. **Project Structure**
   - Establish a clear directory structure based on the technology stack
   - Separate concerns (source, tests, docs, config)
   - Example structures:
     ```
     # Python quantum project
     src/
     tests/
     docs/
     requirements.txt
     setup.py

     # JavaScript/TypeScript quantum project
     src/
     tests/
     docs/
     package.json
     tsconfig.json

     # Rust quantum project
     src/
     tests/
     docs/
     Cargo.toml
     ```

2. **Dependency Management**
   - Add appropriate dependency files (requirements.txt, package.json, Cargo.toml, etc.)
   - Document installation instructions in README.md
   - Pin dependency versions for reproducibility

3. **Testing Infrastructure**
   - Set up testing framework appropriate to the language
   - Add test files alongside source code
   - Include test running instructions in documentation
   - Aim for good test coverage on critical quantum algorithms

4. **Documentation**
   - Expand README.md with:
     - Installation instructions
     - Usage examples
     - API documentation
     - Contributing guidelines
   - Consider adding:
     - CONTRIBUTING.md
     - LICENSE
     - CHANGELOG.md
     - API.md

5. **Configuration Files**
   - Add .gitignore for the chosen technology stack
   - Consider .editorconfig for consistent formatting
   - Add linter/formatter configuration (.eslintrc, .pylintrc, rustfmt.toml, etc.)

6. **CI/CD**
   - Consider adding .github/workflows/ for GitHub Actions
   - Automate testing, linting, building
   - Add status badges to README.md

### Quantum Computing Specific Considerations

If this repository will contain quantum computing code:

1. **Framework Selection**
   - Document which quantum framework is being used (Qiskit, Cirq, Q#, etc.)
   - Specify quantum simulator vs. real hardware targets
   - Document quantum backend requirements

2. **Quantum Circuit Documentation**
   - Add clear comments for quantum gates and circuits
   - Document qubit registers and their purposes
   - Explain measurement strategies

3. **Performance Considerations**
   - Document computational complexity of quantum algorithms
   - Specify required qubit counts
   - Note quantum circuit depth limitations

4. **Validation**
   - Include classical verification of quantum results where possible
   - Document expected outputs and error rates
   - Add benchmarking for quantum algorithm performance

## Communication Style

### For AI Assistants

1. **Tone**
   - Be concise and direct
   - Avoid emojis unless explicitly requested
   - Use professional, objective language
   - Focus on technical accuracy over validation

2. **Output Format**
   - Use GitHub-flavored markdown
   - Format for command-line interface display
   - Keep responses short and to the point
   - Use code blocks for code examples

3. **Code References**
   - Reference specific locations using `file_path:line_number` format
   - Example: "Error handling occurs in src/main.py:42"

## Tool Usage Patterns

### Parallel Tool Calls
When multiple independent operations are needed, call tools in parallel:
- Multiple file reads
- Independent grep/glob searches
- Parallel bash commands (git status, git diff, git log)

### Sequential Tool Calls
Use sequential calls when operations depend on previous results:
- Read file, then edit based on contents
- Search for files, then read matching files
- Commit changes, then push (git add && git commit && git push)

### Specialized Tools
- **Task Tool:** Use for complex, multi-step operations
  - `Explore` subagent: For codebase exploration and understanding
  - `Plan` subagent: For planning implementation strategies
- **WebFetch:** For retrieving and analyzing web content
- **WebSearch:** For current information beyond knowledge cutoff

## Common Workflows

### Adding New Features

1. Create todo list with implementation steps
2. Research existing codebase (use Task/Explore if complex)
3. Implement changes incrementally
4. Test changes (when testing infrastructure exists)
5. Update documentation if needed
6. Commit with descriptive message (when user requests)
7. Push to appropriate branch

### Debugging Issues

1. Reproduce the issue
2. Search codebase for relevant code
3. Read and analyze problematic files
4. Identify root cause
5. Implement fix
6. Verify fix resolves the issue
7. Commit and document the fix

### Code Review

1. Read the modified files
2. Check for:
   - Security vulnerabilities
   - Code quality issues
   - Best practice violations
   - Missing error handling
   - Documentation needs
3. Suggest improvements
4. Implement approved changes

## Resources and References

### Current Documentation
- README.md - Project overview and purpose

### External Resources
- For Claude Code help: https://docs.claude.com/en/docs/claude-code/
- For feedback: https://github.com/anthropics/claude-code/issues

### Repository Metadata
- **Repository Size:** ~151 KB
- **Language:** Not yet specified
- **Framework:** Not yet specified
- **Status:** Initialization/Planning phase

## Change Log

### 2025-11-14
- Created initial CLAUDE.md with comprehensive AI assistant guidelines
- Documented current repository state (early-stage, minimal content)
- Established git workflow and commit conventions
- Added future development recommendations
- Included quantum computing specific considerations

---

**Note to AI Assistants:** This document will evolve as the repository develops. When significant changes are made to the codebase structure, workflows, or conventions, update this file to reflect the current state. Always check this file before beginning work to understand the latest repository context and guidelines.
