# Rules for this repository

- I am the ONLY author and contributor of this repository. Claude must not appear as a contributor.
- Never add "Co-Authored-By" or any other co-author trailer to commit messages.
- Never add "Generated with Claude Code" or similar attribution to commits, the README, code
  comments or anywhere else.
- Never change git user.name or user.email.
- Before committing, show `git config user.name` and `git config user.email`; stop if either is empty.
- Commit first, then `git pull --rebase origin main`, then show `git log --format=fuller -1`, then push.
- Never force-push. Never stage `data/`, `.venv/` or anything covered by `.gitignore`.
- Use `.venv/bin/python` to run scripts.
