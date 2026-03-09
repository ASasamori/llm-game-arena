Feature handling
- After completing a new feature, create a test run using that build. Use mock API keys when necessary, so rate-limits do not get incurred
- 

Cloud Notes
- Currently the PostGres server lives on the Raspberry Pi. On my network, it can be connected via Tailscale with the IP Address of 100.116.74.50

Runner/Job Catches
- For the model architecture, use the runner.py script for now. It has workers that are setup at runtime, and then are essentially ports listening for jobs to come in and execute them based on a queue.

Pushing/commits
- Before you push a commit, make sure that all the github workflows work properly. You can replicate this information by using the `act` command through a docker instance.
- When you create code, make sure that you run `pyright`, `ruff check --fix` and `ruff format` to make sure that the code base is clean as possible