# Deep Research Environment

Web research environment powered by Exa API for searching and fetching content.

## Quick Start

```bash
export EXA_API_KEY="your_exa_api_key"

# Build the Docker image
hud build

# Start hot-reload development server
hud dev

# Run the sample tasks
hud eval tasks.json
```

## Deploy

When you're ready to use this environment in production:

1. Push your code to GitHub
2. Connect your repo at [hud.ai](https://hud.ai/environments/new)
3. Builds will trigger automatically on each push

## Tools

- **setup()** - Initialize environment
- **search(query)** - Search the web using Exa API
- **fetch(url)** - Fetch full content from a URL
- **answer(final_answer)** - Submit the final research answer
- **evaluate(expected_answer)** - Evaluate submitted answer against expected result

## Learn More

For complete documentation on building environments and running evaluations, visit [docs.hud.ai](https://docs.hud.ai).
