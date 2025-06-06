# mentorAI 🤖✨

**mentorAI** is an advanced feedback generation system that transforms raw student assessment data into personalized, insightful, and motivational performance reports. By leveraging a powerful Generative AI model, it provides a deep analysis of a student's strengths and weaknesses, complete with actionable advice and visual data representations.

---

## ⭐ Key Features

* **Personalized AI Feedback**: Goes beyond generic scores to provide students with feedback that is encouraging, detailed, and tailored to their specific performance.
* **In-Depth Data Analysis**: Meticulously parses detailed JSON assessment data, analyzing performance at the subject, chapter, concept, and difficulty level to identify nuanced patterns.
* **Cost-Effective AI**: Intelligently designed to leverage free-tier generative models, making the solution highly accessible and scalable without incurring high API costs.
* **Dynamic Visualizations**: Automatically generates bar charts to visualize subject-wise accuracy, concept performance, and time management, making complex data easy to understand.
* **Professional PDF Reports**: Creates beautifully styled, multi-page PDF reports that are readable, accessible, and ready to be shared with students, parents, or educators.

---

## 🛠️ How It Works

The system follows a four-step pipeline to generate the final feedback report.

### 1. Data Processing

The script begins by loading a student's assessment data from a JSON file. The core `process_data` function rebuilds all performance metrics from the ground up by analyzing the detailed question list. This ensures accuracy and consistency, even if the source file contains summary-level inconsistencies.

### 2. Prompt Engineering Logic

To ensure the AI's response is structured and insightful, a carefully engineered prompt is sent to the model.

* **Persona Assignment**: The AI is instructed to act as an "Insightful Advisor," setting a supportive and motivational tone.
* **Data Injection**: The prompt is populated with a comprehensive mix of high-level summaries and detailed, granular data.
* **Structured Output Requirement**: The model is explicitly commanded to return its entire response as a single, valid JSON object wrapped in a markdown code block (```` ```json ... ``` ````). This is the key to enabling structured data extraction from a text-only output.

### 3. AI Feedback Generation

* **API Used**: The system uses the Google Generative AI API to access a model like `gemma-3-27b-it` from the free tier.
* **Intelligent Parsing**: To work effectively with models that do not support a dedicated JSON output mode, the system requests a plain-text response. **It then intelligently parses this text using regular expressions to reliably locate and extract the structured JSON object from the markdown block.** This makes the solution both cost-effective and robust.

### 4. PDF Report Structure

The final step assembles the processed data, charts, and AI-generated text into a polished PDF report using the `reportlab` library. The report is logically structured to guide the student through their results:

1.  **Title Page**: Student name, assessment date, and subjects.
2.  **Personalized Introduction**: A welcoming and motivational intro from the AI advisor.
3.  **Overall Performance Snapshot**: A summary of overall accuracy, correct answers, and time taken.
4.  **Subject-wise Analysis**: A breakdown of performance in each subject, including a bar chart for visual comparison.
5.  **Chapter-wise Insights**: AI-driven analysis of strong and weak chapters, supported by a detailed data table.
6.  **Concept & Difficulty Insights**: Deeper analysis of specific concepts and performance across difficulty levels, visualized with charts.
7.  **Time Management Insights**: An analysis of the student's pacing, visualized with a chart.
8.  **Actionable Suggestions**: A numbered list of specific, practical steps for improvement.
9.  **Concluding Remark**: A final, uplifting message to motivate the student.

---

## 🏆 Project Highlights

### Cost-Effective AI Integration
The system is intelligently designed to work with free-tier models like Gemma, which do not have a dedicated JSON output mode. By instructing the model to wrap its response in a specific markdown format and **then using regular expressions to parse the plain-text output, we achieve reliable, structured data extraction without relying on premium APIs.** This makes the solution highly accessible and cost-effective.

### Prompt Engineering
The system utilizes a sophisticated prompt that assigns a specific persona ("Insightful Advisor") to the AI, provides a rich mix of summarized and raw statistical data, and enforces a strict output format. This ensures the generated feedback is not only personalized and empathetic but also structured for reliable, automated integration.

### Data Interpretation
Instead of relying on top-level summaries, the script performs a granular analysis of every question attempted by the student. It programmatically derives performance metrics for chapters, topics, concepts, and difficulty levels, allowing it to accurately identify the student’s specific areas of strength and weakness with high precision.

### AI-Generated Output
The feedback generated by the AI is context-aware and highly relevant. By analyzing the detailed data provided, the model produces motivational and constructive narratives that go beyond mere statistics, offering explanations and encouragement tailored to the student's unique performance patterns.

### PDF Report Quality
The final output is a professionally designed, multi-page PDF created with `reportlab`. It features a clean and readable layout, a consistent color scheme, and dynamically generated charts (`matplotlib`) that visually represent key data points, making the report both accessible and engaging.

### Code Quality
The codebase is written in a modular and reusable fashion. Functionality is separated into distinct, documented functions for data processing, API interaction, chart generation, and PDF creation. This makes the code easy to read, maintain, and extend for future enhancements.

---

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### Prerequisites

* Python 3.7+
* pip package installer

### 1. Installation

Clone the repository and install the required packages:

```bash
# Clone the repository
git clone https://github.com/your-username/mentorAI.git
cd mentorAI

# Install dependencies
pip install google-generativeai reportlab matplotlib python-dotenv numpy
```

### 2. Configuration

Create a file named `.env` in the root directory of the project and add your Google Generative AI API key:

```.env
GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

### 3. Running the Script

Execute the script from your terminal, providing the path to the student's JSON data file.

```bash
python feedback_system.py path/to/your/submission_data.json
```

A PDF report will be generated in the project's root directory. If you run the script without providing a file path, it will automatically create and process a `dummy_submission.json` file for demonstration purposes.