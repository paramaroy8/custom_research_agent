# Project: Custom Research Agent

## Description

- Developed a custom AI research assistant capable of gathering, summarizing, and evaluating
information across web and Wikipedia sources
- Integrated OpenAI’s GPT-4 via the LangChain framework to process multi-step research
workflows with dynamic tool usage
- Instrumented full pipeline observability and traceability using the JudgmentLab Tracer and
FaithfulnessScorer to assess agent accuracy and source quality
- Implemented robust fallback evaluation using OpenAI's GPT-based evaluator to ensure scoring
continuity when project quota limits are exceeded
- Enabled logging of intermediate steps and final outputs into timestamped text files for offline
evaluation and making the process auditable

Table of Contents

```
Installation
Setup
Usage
Save Output to File
Requirements
License
```

## Installation

To get started, you'll need to install the required dependencies. You can do this by running:

```
pip install -r requirements.txt
```

## Setup


### Step 1: Create an Account
First, you'll need to create an account on the [Judgment Lab Platform](https://app.judgmentlabs.ai/register). After registration, you'll get the necessary API keys.

### Step 2: Set Up Environment Variables
Create a .env file in your project’s root directory and add the following entries:

```
OPENAI_API_KEY=your_openai_api_key
JUDGMENT_API_KEY=your_judgment_api_key
JUDGMENT_ORG_ID=your_org_id
```
Replace "your_openai_api_key", "your_judgment_api_key", and "your_org_id" with the corresponding values from the OpenAI API and Judgment Lab Platform.

### Step 3: Run the Project
Once everything is set up, you can start the project by running the following command:
```
python main.py
```

## Usage

Once the script is running, you can interact with the agent by providing it with the following inputs:

```
  • Your Question/Topic – What would you like to know more about? This could be any topic or query.
  • Maximum Number of Steps – Define how many steps you want the agent to take to retrieve information.
  • Evaluation Option – Choose whether you want to evaluate the agent's steps. Type y for Yes and n for No.
```
  
Example:
```
What is Quantum Mechanics?
5
y
```
This would prompt the agent to look for information on the topic "What is Quantum Mechanics?" and execute up to 5 steps, asking you to evaluate each step.

### Save Output to File

If you want to save the agent's steps to a text file in the current directory, simply type "save to file" after entering your query. The agent will output each step into a file named agent_output.txt.

Example:
```
What is Quantum Mechanics? save to file
```
This will cause the agent to save all the steps it took to gather information into the file agent_output.txt.

## Requirements

```
Python 3.10 (or, higher)
```
Dependencies listed in requirements.txt

