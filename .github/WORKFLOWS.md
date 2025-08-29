# GitHub Workflows Documentation

This document describes the GitHub Actions workflows used in the Quantra repository.

## Auto-Assign Copilot as Reviewer

This workflow automatically assigns GitHub Copilot as a reviewer on AI-generated or AI-related pull requests.

### Workflow File
`.github/workflows/assign-copilot-reviewer.yml`

### Trigger Events
- Pull request labeled with AI-related labels
- Review requested on a pull request
- Pull request opened or reopened
- Manual trigger via workflow_dispatch (requires PR number)

### How It Works
1. The workflow runs when a PR is opened, reopened, labeled, or has a review requested
2. It checks if the PR is AI-related through any of these signals:
   - Has AI-related labels (e.g., 'review-requested-by-ai', 'ai-generated', 'copilot')
   - PR title contains AI-related keywords
   - PR description contains AI-related keywords
   - PR was created by a bot or AI-related account
3. If the PR is determined to be AI-related, GitHub Copilot is assigned as a reviewer

### Configuration
The workflow can be configured by:
1. Adding or removing AI-related labels in the `aiRelatedLabels` array
2. Modifying the AI-related keywords in the `aiKeywords` array  
3. Changing the AI author accounts in the `aiAuthors` array

### Manual Trigger
You can manually trigger this workflow for a specific PR by:
1. Go to Actions > Auto-Assign Copilot as Reviewer > Run workflow
2. Enter the PR number
3. Click "Run workflow"

## Test Copilot Reviewer Assignment

This workflow is used to test the detection logic of the Auto-Assign Copilot as Reviewer workflow without actually assigning reviewers.

### Workflow File
`.github/workflows/test-copilot-reviewer.yml`

### Trigger Events
- Manual trigger only via workflow_dispatch

### How It Works
1. The workflow creates a mock PR based on the input parameters
2. It then runs the same detection logic used in the main workflow
3. The workflow reports whether the mock PR would be detected as AI-related
4. This allows testing different combinations of PR titles, bodies, and labels

### Manual Testing
1. Go to Actions > Test Copilot Reviewer Assignment > Run workflow
2. Enter test values for PR title, body, and labels
3. Click "Run workflow"
4. Check the workflow run to see if the PR would be detected correctly

## Auto-Assign

This workflow automatically assigns the repository owner to new issues and PRs.

### Workflow File
`.github/workflows/auto-assign.yml`

### Trigger Events
- New issue opened
- New pull request opened

## Proof HTML

This workflow validates HTML in the repository.

### Workflow File
`.github/workflows/proof-html.yml`

### Trigger Events
- Any push to the repository
- Manual trigger via workflow_dispatch