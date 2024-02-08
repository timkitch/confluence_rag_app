# Q&A Bot for Confluence Page

This project is a Q&A bot for Confluence pages.

The UI allows you to ingest a Confluence space in two ways:

1. You can ingest specific pages by providing page ids.
2. You can ingest the entire Confluence space by leaving the "Page ID" field blank.

## Getting Started

These instructions will help you get the project up and running on your local machine.

The bot currently uses OpenAI's GPT 3.5 Turbo model. The model to use is defined in the ```constants.py``` file's ```LLM``` variable. You can change this variable to point to any OpenAI LLM.

Before getting started, create an OpenAI API key by going to https://platform.openai.com/api-keys. Then, create a file named ```.env``` in the root project directory and add this line:

```
OPENAI_API_KEY=<your_API_key>
```

If you don't have a Confluence API key, log into the Atlassian Confluence wiki you want to ingest and perform Q&A against and generate an API key. This page is typically: https://id.atlassian.com/manage-profile/security/api-tokens. Copy the value of this key to a safe location, as you will not be able to retrieve it via the Confluence web app later. You will need this key once you launch the Q&A bot.

### Prerequisites

This project requires Python and pip. You can install the required packages with the following command:

```sh
pip install -r requirements.txt

./run.sh
```

