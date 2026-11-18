
<div align="center">

# 📚 AI-Powered Literature Review and Research Analysis System

### An Intelligent Multi-Agent System for Automated Academic Research and Literature Analysis

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Flask-Backend-black?logo=flask" alt="Flask">
  <img src="https://img.shields.io/badge/LangGraph-Agentic%20AI-green" alt="LangGraph">
  <img src="https://img.shields.io/badge/NLP-Research%20Analysis-orange" alt="NLP">
  <img src="https://img.shields.io/badge/Status-Active-success" alt="Status">
</p>

**Automating research discovery, synthesis, fact verification, hypothesis generation, and research evaluation.**

</div>

---

## 📌 Overview

The **AI-Powered Literature Review and Research Analysis System** is an intelligent research assistance application developed to simplify and automate the academic literature review process.

The application uses a multi-agent architecture orchestrated through **LangGraph**, where specialized agents collaborate to retrieve academic research papers, synthesize findings, verify factual consistency, identify research gaps, generate hypotheses, detect potentially similar published research, and evaluate research quality.

The system integrates the **OpenAlex API** for academic research retrieval and locally executed NLP and Machine Learning models for research analysis.

It also provides an interactive desktop interface, research history management, side-by-side literature comparison, and downloadable research reports in Word and PDF formats.

The goal is to make academic research more structured, efficient, and evidence-oriented.

## 🎯 Project Objectives

- Automate the academic literature review workflow.
- Retrieve relevant research papers from academic sources.
- Organize research papers into meaningful themes.
- Generate concise summaries of research findings.
- Verify generated summaries against source abstracts.
- Identify research gaps and potential contradictions.
- Generate structured research hypotheses.
- Analyze the similarity of hypotheses with existing publications.
- Evaluate the quality and coverage of research outputs.
- Generate downloadable research reports.

## ✨ Key Features

### 1. Academic Research Paper Retrieval

- Retrieves academic papers using the OpenAlex API.
- Extracts available paper information, including titles, abstracts, authors, and publication years.
- Removes duplicate research papers.
- Filters papers based on relevance.
- Supports publication year and open-access filtering.
- Uses local caching to reduce repeated API requests.

### 2. Intelligent Research Synthesis

- Groups research papers into themes using KMeans clustering.
- Generates extractive summaries using the TextRank algorithm.
- Identifies common research themes.
- Highlights potential research gaps.
- Identifies possible contradictions between research themes.

### 3. AI-Based Fact Verification

- Uses a Natural Language Inference (NLI) model.
- Compares generated summary sentences against original research abstracts.
- Evaluates entailment, contradiction, and neutral evidence.
- Identifies unsupported research statements.
- Uses an iterative self-refinement process.
- Calculates a final support ratio for the synthesized findings.

### 4. Research Hypothesis Generation

- Identifies meaningful research terms using TF-IDF.
- Uses research gaps and low-coverage themes.
- Generates structured and testable research hypotheses.
- Supports configurable hypothesis generation.

### 5. Research Novelty Detection

- Searches OpenAlex for research related to generated hypotheses.
- Uses semantic embeddings to compare hypotheses with existing publications.
- Identifies potentially similar published research.
- Calculates similarity scores.
- Reports close matches for further investigation.

*Novelty detection is similarity-based and does not guarantee that a research idea is globally original.*

### 6. Research Quality Evaluation

The system evaluates research outputs using multiple indicators:

- **Coverage:** Measures how well retrieved papers are represented in the synthesis.
- **Hypothesis Specificity:** Evaluates whether generated hypotheses are concrete and testable.
- **Support:** Measures the proportion of synthesis statements supported by source abstracts.
- **Novelty:** Evaluates hypotheses based on their similarity to retrieved published research.
- **Overall Score:** Combines available evaluation indicators into an overall quality measure.

### 7. Research Comparison

- Compare two completed literature review searches.
- Identify common and unique research papers.
- Compare important research terms.
- Analyze theme similarity using cosine similarity.
- Generate a comparison summary.

### 8. Search History Management

- Automatically saves completed research runs.
- Stores research topics, settings, reports, and evaluation results.
- Reopens previously completed searches.
- Deletes individual history entries.
- Supports clearing saved search history.

### 9. Research Report Export

- Export research reports in Microsoft Word format (.docx).
- Export research reports in PDF format.
- Include research themes, findings, hypotheses, novelty results, and references.
- Store generated reports locally.

---

## 🏗️ System Architecture

<div align="center">

**Multi-Agent Research Processing Architecture**

<br>

<table>
<tr>
<td align="center" bgcolor="#1f6feb">
<b>📝 Research Topic Input</b>
</td>
</tr>
<tr>
<td align="center">⬇</td>
</tr>
<tr>
<td align="center" bgcolor="#0969da">
<b>🔍 1. Retriever Agent</b>
<br>
<sub>Retrieves and filters academic research papers using OpenAlex</sub>
</td>
</tr>
<tr>
<td align="center">⬇</td>
</tr>
<tr>
<td align="center" bgcolor="#0969da">
<b>📖 2. Synthesizer Agent</b>
<br>
<sub>Groups papers into themes and generates summaries</sub>
</td>
</tr>
<tr>
<td align="center">⬇</td>
</tr>
<tr>
<td align="center" bgcolor="#0969da">
<b>🛡️ 3. Critic Agent</b>
<br>
<sub>Verifies findings and refines unsupported statements</sub>
</td>
</tr>
<tr>
<td align="center">⬇</td>
</tr>
<tr>
<td align="center" bgcolor="#0969da">
<b>💡 4. Hypothesis Generator</b>
<br>
<sub>Generates hypotheses from research gaps</sub>
</td>
</tr>
<tr>
<td align="center">⬇</td>
</tr>
<tr>
<td align="center" bgcolor="#0969da">
<b>🔬 5. Novelty Checker</b>
<br>
<sub>Compares hypotheses with existing publications</sub>
</td>
</tr>
<tr>
<td align="center">⬇</td>
</tr>
<tr>
<td align="center" bgcolor="#0969da">
<b>📊 6. Evaluator Agent</b>
<br>
<sub>Evaluates research coverage and hypothesis quality</sub>
</td>
</tr>
<tr>
<td align="center">⬇</td>
</tr>
<tr>
<td align="center" bgcolor="#238636">
<b>📑 Final Research Report</b>
</td>
</tr>
<tr>
<td align="center">⬇</td>
</tr>
<tr>
<td align="center">
<b>Research History | Word Export | PDF Export | Comparison</b>
</td>
</tr>
</table>

<br>

<sub>Workflow orchestration: LangGraph | Research retrieval: OpenAlex | Analysis: Local NLP and ML models</sub>

</div>

### Multi-Agent Workflow

<div align="center">

<table>
<thead>
<tr>
<th align="center">Stage</th>
<th align="center">Agent</th>
<th align="center">Responsibility</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">01</td>
<td align="center">Retriever</td>
<td align="center">Retrieves and filters academic papers</td>
</tr>
<tr>
<td align="center">02</td>
<td align="center">Synthesizer</td>
<td align="center">Groups papers into themes and summarizes findings</td>
</tr>
<tr>
<td align="center">03</td>
<td align="center">Critic</td>
<td align="center">Verifies summaries against source abstracts</td>
</tr>
<tr>
<td align="center">04</td>
<td align="center">Hypothesis Generator</td>
<td align="center">Generates hypotheses from research gaps</td>
</tr>
<tr>
<td align="center">05</td>
<td align="center">Novelty Checker</td>
<td align="center">Checks for similar published research</td>
</tr>
<tr>
<td align="center">06</td>
<td align="center">Evaluator</td>
<td align="center">Calculates research quality indicators</td>
</tr>
</tbody>
</table>

</div>

## 🛠️ Technology Stack

<div align="center">

<table>
<thead>
<tr>
<th align="center">Category</th>
<th align="center">Technology</th>
</tr>
</thead>
<tbody>
<tr><td align="center">Programming Language</td><td align="center">Python</td></tr>
<tr><td align="center">Backend Framework</td><td align="center">Flask</td></tr>
<tr><td align="center">Desktop Interface</td><td align="center">PyWebView</td></tr>
<tr><td align="center">Agent Orchestration</td><td align="center">LangGraph</td></tr>
<tr><td align="center">Academic Research API</td><td align="center">OpenAlex API</td></tr>
<tr><td align="center">Natural Language Processing</td><td align="center">NLP</td></tr>
<tr><td align="center">Text Embeddings</td><td align="center">Sentence Transformers</td></tr>
<tr><td align="center">Embedding Model</td><td align="center">all-MiniLM-L6-v2</td></tr>
<tr><td align="center">NLI Model</td><td align="center">cross-encoder/nli-deberta-v3-base</td></tr>
<tr><td align="center">Machine Learning</td><td align="center">Scikit-learn</td></tr>
<tr><td align="center">Clustering</td><td align="center">KMeans</td></tr>
<tr><td align="center">Text Summarization</td><td align="center">TextRank</td></tr>
<tr><td align="center">Feature Extraction</td><td align="center">TF-IDF</td></tr>
<tr><td align="center">Similarity Measurement</td><td align="center">Cosine Similarity</td></tr>
<tr><td align="center">Numerical Processing</td><td align="center">NumPy</td></tr>
<tr><td align="center">Graph Processing</td><td align="center">NetworkX</td></tr>
<tr><td align="center">HTTP Requests</td><td align="center">Requests</td></tr>
<tr><td align="center">Document Generation</td><td align="center">python-docx, fpdf2</td></tr>
<tr><td align="center">Environment Configuration</td><td align="center">python-dotenv</td></tr>
<tr><td align="center">Progress Display</td><td align="center">tqdm</td></tr>
<tr><td align="center">Frontend</td><td align="center">HTML, CSS, JavaScript</td></tr>
</tbody>
</table>

</div>

## 📂 Project Structure

```text
ai-powered-literature-review-system/
│
├── app.py
├── config.py
│
├── graph_pipeline.py
├── pipeline.py
│
├── retriever.py
├── synthesizer.py
├── critic.py
├── evaluator.py
├── hypothesis_generator.py
├── novelty_checker.py
│
├── vector_store.py
├── textrank.py
├── comparison.py
├── history.py
├── export.py
│
├── calculate_metrics.py
├── error_injection.py
├── check_links.py
│
├── experiment_summary.json
│
├── static/
│   └── index.html
│
├── exports/
│
├── history/
│
├── test_comparison.py
├── test_evaluator.py
├── test_graph_pipeline.py
├── test_history.py
│
├── .env.example
├── .gitignore
└── README.md
```

### Important Modules

<div align="center">

<table>
<thead>
<tr>
<th align="center">File</th>
<th align="center">Purpose</th>
</tr>
</thead>
<tbody>
<tr><td align="center">app.py</td><td align="center">Main Flask and PyWebView application</td></tr>
<tr><td align="center">config.py</td><td align="center">Configuration, model settings, and directories</td></tr>
<tr><td align="center">graph_pipeline.py</td><td align="center">LangGraph multi-agent workflow</td></tr>
<tr><td align="center">pipeline.py</td><td align="center">Command-line pipeline execution</td></tr>
<tr><td align="center">retriever.py</td><td align="center">Academic research retrieval</td></tr>
<tr><td align="center">synthesizer.py</td><td align="center">Research synthesis and thematic summaries</td></tr>
<tr><td align="center">critic.py</td><td align="center">Fact verification and self-refinement</td></tr>
<tr><td align="center">hypothesis_generator.py</td><td align="center">Research hypothesis generation</td></tr>
<tr><td align="center">novelty_checker.py</td><td align="center">Research novelty and similarity checking</td></tr>
<tr><td align="center">evaluator.py</td><td align="center">Research quality evaluation</td></tr>
<tr><td align="center">vector_store.py</td><td align="center">Embedding generation and vector storage</td></tr>
<tr><td align="center">textrank.py</td><td align="center">Extractive text summarization</td></tr>
<tr><td align="center">comparison.py</td><td align="center">Literature review comparison</td></tr>
<tr><td align="center">history.py</td><td align="center">Research history management</td></tr>
<tr><td align="center">export.py</td><td align="center">Word and PDF report generation</td></tr>
<tr><td align="center">calculate_metrics.py</td><td align="center">Experimental evaluation and metrics</td></tr>
</tbody>
</table>

</div>

## ⚙️ Installation and Setup

### Prerequisites

Make sure you have installed:

- Python 3.10 or above
- pip
- Git
- OpenAlex API key
- Internet connection for research retrieval and initial model downloads

### Step 1: Clone the Repository

```bash
git clone https://github.com/khushi-io/ai-powered-literature-review-system.git
```

### Step 2: Navigate to the Project Directory

```bash
cd ai-powered-literature-review-system
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### Step 5: Install Dependencies

```bash
pip install flask pywebview langgraph requests numpy sentence-transformers scikit-learn networkx python-dotenv tqdm python-docx fpdf2
```

### Step 6: Configure Environment Variables

Create a `.env` file in the root directory.

Add your OpenAlex API key:

```env
OPENALEX_API_KEY=your_openalex_api_key
```

Replace the placeholder with your actual API key.

**Important:** Never upload your real `.env` file or API key to GitHub.

### Step 7: Run the Application

```bash
python app.py
```

The application starts a local Flask server and launches the desktop interface using PyWebView.

## 💻 Command-Line Usage

The project also supports direct execution of the literature review pipeline through the command line.

### Run a Research Topic

```bash
python pipeline.py "machine learning for crop yield prediction"
```

### Force a Fresh Research Retrieval

```bash
python pipeline.py "deep learning for medical image segmentation" --fresh
```

### Customize Research Settings

```bash
python pipeline.py "machine learning for crop yield prediction" --max-papers 10 --min-relevance 0
```

### Filter by Publication Year

```bash
python pipeline.py "artificial intelligence in healthcare" --year-from 2020 --year-to 2026
```

### Retrieve Open Access Papers

```bash
python pipeline.py "natural language processing" --open-access
```

### Customize the Number of Hypotheses

```bash
python pipeline.py "machine learning in education" --num-hypotheses 5
```

### Display Pipeline Diagram Text

```bash
python pipeline.py --diagram
```

## 🧪 Testing

The project includes test modules to validate important application components.

### Run All Tests

```bash
python -m pytest
```

### Run Individual Test Modules

```bash
python -m pytest test_comparison.py
```

```bash
python -m pytest test_evaluator.py
```

```bash
python -m pytest test_graph_pipeline.py
```

```bash
python -m pytest test_history.py
```

## 📊 Experimental Evaluation

The project includes an experimental evaluation module to assess the effectiveness of the Critic agent's self-refinement process.

### Experimental Conditions

**Condition A – Baseline**

- Uses deliberately corrupted research synthesis.
- Checks the synthesis without iterative self-refinement.
- Measures the resulting support ratio.

**Condition B – Critic Self-Refinement**

- Uses the same deliberately corrupted synthesis.
- Applies the Critic verification and revision process.
- Measures the support ratio after refinement.

### Error Injection Techniques

- Negation injection
- Number swapping
- Sentence splicing

The experiment compares the results to examine how the Critic agent improves factual support in generated research summaries.

### Run the Experiment

```bash
python calculate_metrics.py
```

## 🔐 Implementation Notes

- LangGraph coordinates the multi-agent research workflow.
- OpenAlex provides the academic research retrieval service.
- Sentence Transformers generates semantic embeddings.
- The NLI model supports evidence verification.
- KMeans is used for thematic clustering.
- TextRank supports extractive summarization.
- TF-IDF supports research term extraction.
- Cosine similarity supports semantic comparison.
- Search history, exports, and caches are stored locally.
- Novelty results are similarity indicators, not guarantees of originality.
- Evaluation scores should be interpreted as research-assistance indicators, not proof of scientific validity.

## 🚀 Future Enhancements

- Integrate additional academic research databases.
- Improve research paper retrieval and ranking.
- Enhance research gap identification.
- Improve hypothesis generation and evaluation.
- Introduce advanced research visualization.
- Expand citation and reference management.
- Add more automated testing and validation.
- Improve application performance and scalability.


---

<div align="center">

## 👩‍💻 Author

### Khushi Chile


## 💡 A Thought Behind This Project

**"Research is not just about finding answers; it's about questioning what exists, discovering what is missing, and creating possibilities for what comes next."**

<br>

**AI-Powered Literature Review and Research Analysis System**

*Transforming academic research through intelligent automation.*

<br>

⭐ **If you find this project useful, don't forget to give it a star!**

</div>
