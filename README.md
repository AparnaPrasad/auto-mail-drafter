# 📬 Auto Mail Drafter

**Auto Mail Drafter** is an intelligent email automation system designed to streamline your email workflow. It automatically fetches incoming emails, evaluates them against predefined rules to determine if a response is necessary, and, if so, drafts a reply. When additional information is required, it retrieves relevant data from your company's documentation stored in a Pinecone vector database.

---

## 🧠 Features

- **Automated Email Retrieval**: Fetches emails from Gmail
- **Rule-Based Decision Engine**: Determines whether an email requires a response using configurable business rules.
- **ReAact (Reasoning + Action) based Decision:** If a reply is needed, it uses Reasoning to decide if more information is needed to answer the email.
- **Dynamic Information Lookup**: If more information is needed, retrieves supplementary information from company documents stored in Pinecone.
- **Context-Aware Reply Generation**: Uses a language model to draft responses tailored to the conversation.
- **Human-in-the-Loop Workflow**: Enables manual review and editing of draft responses before sending.

---

## 🛠️ Installation

### Backend (Python + Pipenv)

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/AparnaPrasad/auto-mail-drafter.git
   cd auto-mail-drafter
   ```
2. **Install Pipenv** (if not already installed):
  ```bash
  pip install pipenv
  ```
3. **Install Dependencies:**
   ```bash
   pipenv install
   ```
4. **Create .env File:**
   Add API keys to .env file
5. **Run the agent**
   ```bash
   python agents/email-agent.py
   ```
   
## ⚙️ Configuration
You can adjust the following:
- Rules: Define custom logic in config.py for deciding whether an email needs a reply.
- LLM Prompt Templates: Customize prompts used to generate email responses in agents/email-agents.py

## 🤝 Contributing
Contributions are welcome and appreciated!

Fork the repo

Create a feature branch

Submit a pull request

Please include relevant tests and a clear explanation of your changes.

## 📄 License
This project is licensed under the MIT License.

## 🙋‍♀️ Maintainer
Developed by Aparna Prasad
