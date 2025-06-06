import json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib import colors
from datetime import datetime
import textwrap
import os
import re
from dotenv import load_dotenv

# For charts
import matplotlib.pyplot as plt
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Attempt to import Gemini; handle if not installed
try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: 'google-generativeai' library not found. LLM calls will be mocked. Please install with 'pip install google-generativeai'")


# --- Configuration & Styles ---
styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    'TitleStyle',
    parent=styles['h1'],
    fontSize=22,
    alignment=TA_CENTER,
    spaceAfter=20,
    textColor=colors.HexColor("#4A90E2")
)
heading1_style = ParagraphStyle(
    'Heading1Style',
    parent=styles['h2'],
    fontSize=16,
    spaceBefore=12,
    spaceAfter=6,
    textColor=colors.HexColor("#34495E")
)
heading2_style = ParagraphStyle(
    'Heading2Style',
    parent=styles['h3'],
    fontSize=14,
    spaceBefore=10,
    spaceAfter=4,
    textColor=colors.HexColor("#2C3E50")
)
body_style = ParagraphStyle(
    'BodyStyle',
    parent=styles['Normal'],
    fontSize=11,
    leading=14,
    alignment=TA_JUSTIFY,
    spaceAfter=10,
    textColor=colors.HexColor("#333333")
)
bullet_style = ParagraphStyle(
    'BulletStyle',
    parent=body_style,
    leftIndent=20,
    bulletIndent=10,
    spaceAfter=5
)
code_style = ParagraphStyle(
    'CodeStyle',
    parent=styles['Code'],
    fontSize=10,
    leading=12,
    textColor=colors.darkgrey
)

# --- 1. Data Processing ---
def load_data(json_file_path):
    """Loads data from a JSON file, specifying UTF-8 encoding."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded data from {json_file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {json_file_path}. Details: {e}")
        return None

def extract_subject_names_from_syllabus(syllabus_html):
    """Extracts subject names from syllabus HTML using a regex."""
    if not syllabus_html:
        return []
    subject_names = re.findall(r'<h2>(.*?)<\/h2>', syllabus_html)
    return [name.strip() for name in subject_names]

def process_data(data):
    """
    Processes raw assessment data from the question level to ensure accuracy and detail,
    overriding any inconsistent top-level summaries. It handles structures like in
    'sample_submission_analysis_1.json' by parsing the 'sections' array.
    """
    if not data:
        return None

    processed = {}
    
    # Safely get studentName, handling potential nested structures or using _id
    processed['studentName'] = data.get('studentName')
    if not processed['studentName'] and 'test' in data and 'studentName' in data['test']:
        processed['studentName'] = data['test']['studentName']
    if not processed['studentName'] and '_id' in data and '$oid' in data['_id']:
        processed['studentName'] = f"Student_{data['_id']['$oid'][-4:]}" # Last 4 chars of OID
    if not processed['studentName']:
        processed['studentName'] = 'Unknown Student'
        
    processed['assessmentDate'] = data.get('assessmentDate', datetime.now().strftime("%Y-%m-%d"))

    # Dictionaries to aggregate fresh data from the question list
    subject_performance = {}
    chapter_performance = {}
    concept_performance = {}
    difficulty_performance = {
        'Easy': {'correct': 0, 'total': 0, 'time': 0},
        'Medium': {'correct': 0, 'total': 0, 'time': 0},
        'Hard': {'correct': 0, 'total': 0, 'time': 0},
        'Tough': {'correct': 0, 'total': 0, 'time': 0},
        'Unknown': {'correct': 0, 'total': 0, 'time': 0}
    }

    time_correct_q_sec, time_incorrect_q_sec = 0, 0
    num_correct_q, num_incorrect_q = 0, 0
    overall_total_q, overall_correct_q, overall_total_time_sec = 0, 0, 0

    # Get subject names from syllabus to map to sections
    syllabus_html = data.get('test', {}).get('syllabus', '')
    subject_names_from_syllabus = extract_subject_names_from_syllabus(syllabus_html)

    # Flatten all questions from all sections into a single list for processing
    all_questions = []
    
    # Primary structure: 'sections' array (from sample_submission_analysis_1.json)
    if 'sections' in data:
        for section in data.get('sections', []):
            section_title = section.get('sectionId', {}).get('title', 'Unknown Section')
            # Infer subject from section title for all questions in this section
            current_subject = 'Unknown'
            for sub_name in subject_names_from_syllabus:
                if sub_name.lower() in section_title.lower():
                    current_subject = sub_name
                    break
            
            for q_data in section.get('questions', []):
                q_data['inferred_subject'] = current_subject
                all_questions.append(q_data)
                
    # Fallback structure: 'chapters' array with nested 'questions' (from old dummy data)
    elif 'chapters' in data:
        for chapter_group in data.get('chapters', []):
            for q_data in chapter_group.get('questions', []):
                all_questions.append(q_data)

    # Heuristic map for subject inference if not found via section titles
    chapter_to_subject_map = {
        "Electrostatics": "Physics", "Capacitance": "Physics",
        "Solutions": "Chemistry", "Electrochemistry": "Chemistry",
        "Sets and Relations": "Maths", "Functions": "Maths"
    }
    
    # Iterate through the aggregated list of all questions
    for q_data in all_questions:
        status = q_data.get('status')
        # Only process questions that were actually attempted
        if status not in ['answered', 'markedReview']:
            continue

        # Robustly determine if the answer was correct, handling different JSON formats
        is_correct = False
        if q_data.get('inputValue', {}).get('isCorrect') is not None:
            is_correct = q_data['inputValue']['isCorrect']
        elif 'markedOptions' in q_data and q_data['markedOptions']:
            is_correct = any(opt.get('isCorrect', False) for opt in q_data['markedOptions'])

        time_taken = q_data.get('timeTaken', 0)
        
        # Get details from the 'questionId' block
        q_id_data = q_data.get('questionId', {})
        difficulty = q_id_data.get('level', 'Unknown').capitalize()
        chapters = [ch.get('title') for ch in q_id_data.get('chapters', []) if ch.get('title')]
        if not chapters: chapters = ["Uncategorized"]
        
        # Merge 'topics' and 'concepts' into a single list for analysis
        concepts = [t.get('title') for t in q_id_data.get('topics', []) if t.get('title')]
        concepts.extend([c.get('title') for c in q_id_data.get('concepts', []) if c.get('title')])
        if not concepts: concepts = ["Uncategorized"]

        # --- Aggregate Data Points ---

        # 1. Overall stats
        overall_total_q += 1
        overall_total_time_sec += time_taken
        if is_correct:
            overall_correct_q += 1
            time_correct_q_sec += time_taken
            num_correct_q += 1
        else:
            time_incorrect_q_sec += time_taken
            num_incorrect_q += 1
            
        # 2. Subject-level stats
        subject_name = q_data.get('inferred_subject', 'Unknown')
        if subject_name == 'Unknown':
            for ch in chapters:
                if ch in chapter_to_subject_map:
                    subject_name = chapter_to_subject_map[ch]
                    break
        
        if subject_name not in subject_performance:
            subject_performance[subject_name] = {'totalQuestions': 0, 'correctAnswers': 0, 'totalTimeTakenSeconds': 0}
        subject_performance[subject_name]['totalQuestions'] += 1
        subject_performance[subject_name]['totalTimeTakenSeconds'] += time_taken
        if is_correct:
            subject_performance[subject_name]['correctAnswers'] += 1

        # 3. Chapter-level stats
        for chapter in chapters:
            if chapter not in chapter_performance:
                chapter_performance[chapter] = {'totalQuestions': 0, 'correctAnswers': 0, 'timeTakenSeconds': 0, 'subjectName': subject_name}
            chapter_performance[chapter]['totalQuestions'] += 1
            chapter_performance[chapter]['timeTakenSeconds'] += time_taken
            if is_correct:
                chapter_performance[chapter]['correctAnswers'] += 1
        
        # 4. Concept-level stats
        for concept in concepts:
            if concept not in concept_performance:
                concept_performance[concept] = {'correct': 0, 'total': 0, 'time': 0}
            concept_performance[concept]['total'] += 1
            concept_performance[concept]['time'] += time_taken
            if is_correct:
                concept_performance[concept]['correct'] += 1

        # 5. Difficulty-level stats
        diff_cat = difficulty if difficulty in difficulty_performance else 'Unknown'
        difficulty_performance[diff_cat]['total'] += 1
        difficulty_performance[diff_cat]['time'] += time_taken
        if is_correct:
            difficulty_performance[diff_cat]['correct'] += 1

    # --- Finalize the 'processed' dictionary with aggregated data ---

    processed['totalQuestions'] = overall_total_q
    processed['correctAnswers'] = overall_correct_q
    processed['overallAccuracy'] = (overall_correct_q / overall_total_q * 100) if overall_total_q > 0 else 0
    processed['totalAssessmentTimeSeconds'] = overall_total_time_sec

    # Finalize Subject data
    processed['subjects'] = []
    for name, stats in subject_performance.items():
        total_q = stats['totalQuestions']
        correct_q = stats['correctAnswers']
        processed['subjects'].append({
            'subjectName': name,
            'totalQuestions': total_q,
            'correctAnswers': correct_q,
            'accuracy': (correct_q / total_q * 100) if total_q > 0 else 0,
            'totalTimeTakenSeconds': stats['totalTimeTakenSeconds']
        })
    processed['subjects'].sort(key=lambda x: x['subjectName'])
    processed['subject_list_str'] = ', '.join([s['subjectName'] for s in processed['subjects']]) if processed['subjects'] else 'General Assessment'

    # Finalize detailed Chapter data
    processed['detailed_chapters'] = []
    for name, stats in chapter_performance.items():
        total_q = stats['totalQuestions']
        correct_q = stats['correctAnswers']
        processed['detailed_chapters'].append({
            'name': name,
            'subjectName': stats['subjectName'],
            'accuracy': (correct_q / total_q * 100) if total_q > 0 else 0,
            'totalQuestions': total_q,
            'correctAnswers': correct_q,
            'timeTakenSeconds': stats['timeTakenSeconds']
        })
    processed['detailed_chapters'].sort(key=lambda x: (x['subjectName'], x['name']))

    # Finalize Concept performance data
    processed['conceptPerformance'] = [{'concept': c, 'accuracy': round((s['correct']/s['total']*100) if s['total']>0 else 0,2), 'correct':s['correct'],'total':s['total'], 'avgTimeSeconds':round(s['time']/s['total'] if s['total']>0 else 0,2)} for c,s in concept_performance.items()]
    
    # Finalize Difficulty performance data
    processed['difficultyPerformance'] = [{'level':d, 'accuracy':round((s['correct']/s['total']*100) if s['total']>0 else 0,2), 'correct':s['correct'],'total':s['total'], 'avgTimeSeconds':round(s['time']/s['total'] if s['total']>0 else 0,2)} for d,s in difficulty_performance.items() if s['total']>0]

    # Finalize Time Analysis data
    processed['timeAnalysis'] = {
        'avgTimePerCorrectQuestionSec': (time_correct_q_sec/num_correct_q) if num_correct_q>0 else 0,
        'avgTimePerIncorrectQuestionSec': (time_incorrect_q_sec/num_incorrect_q) if num_incorrect_q>0 else 0,
        'numCorrect': num_correct_q,
        'numIncorrect': num_incorrect_q,
        'totalTimeCorrectQuestions': time_correct_q_sec,
        'totalTimeIncorrectQuestions': time_incorrect_q_sec
    }

    print("Data processing complete based on detailed question-level analysis.")
    return processed


# --- 2. AI Generated Feedback (Prompt Engineering & LLM Call) ---
def format_data_for_llm(processed_data):
    """Formats the processed data and detailed instructions for the LLM to generate JSON feedback."""
    if not processed_data: return "No data available for analysis."
    
    student_name = processed_data.get('studentName', 'Student')
    assessment_date = processed_data.get('assessmentDate', 'N/A')
    overall_accuracy = processed_data.get('overallAccuracy', 0)
    total_questions = processed_data.get('totalQuestions', 0)
    correct_answers = processed_data.get('correctAnswers', 0)
    total_time_minutes = processed_data.get('totalAssessmentTimeSeconds', 0) / 60

    subject_summaries_for_llm = [
        f"  - {sub['subjectName']}: Accuracy of {sub['accuracy']:.1f}% ({sub['correctAnswers']}/{sub['totalQuestions']}) in {sub['totalTimeTakenSeconds']/60:.1f} minutes."
        for sub in processed_data.get('subjects', [])
    ]
    subjects_text = "\n".join(subject_summaries_for_llm) if subject_summaries_for_llm else "No subject summary data available."

    detailed_chapter_summaries_for_llm = []
    if processed_data.get('detailed_chapters'):
        detailed_chapter_summaries_for_llm.append("\nDetailed Chapter Performance:")
        for ch in processed_data['detailed_chapters']:
             detailed_chapter_summaries_for_llm.append(
                 f"  - Subject: {ch['subjectName']}, Chapter: {ch['name']}: {ch['accuracy']:.1f}% ({ch['correctAnswers']}/{ch['totalQuestions']}) in {ch['timeTakenSeconds']/60:.1f} min"
             )
    detailed_chapters_text = "\n".join(detailed_chapter_summaries_for_llm)

    strong_concepts = sorted([c for c in processed_data.get('conceptPerformance', []) if c['accuracy'] >= 75 and c['total'] > 1], key=lambda x: x['accuracy'], reverse=True)
    weak_concepts = sorted([c for c in processed_data.get('conceptPerformance', []) if c['accuracy'] < 60 and c['total'] > 0], key=lambda x: x['accuracy'])
    
    concepts_text = ""
    if strong_concepts:
        concepts_text += f"Strong concepts: {', '.join([c['concept'] for c in strong_concepts[:5]])}{'...' if len(strong_concepts) > 5 else ''}. "
    if weak_concepts:
        concepts_text += f"Concepts needing review: {', '.join([c['concept'] for c in weak_concepts[:5]])}{'...' if len(weak_concepts) > 5 else ''}."
    if not concepts_text:
        concepts_text = "No specific concept performance insights available."

    difficulty_summary_parts = [f"{d['level']}: {d['accuracy']:.1f}% ({d['correct']}/{d['total']})" for d in processed_data.get('difficultyPerformance', [])]
    difficulty_text = ", ".join(difficulty_summary_parts) if difficulty_summary_parts else "No difficulty-level data available."
    
    ta = processed_data.get('timeAnalysis', {})
    time_analysis_text = (
        f"Average time per correct question: {ta.get('avgTimePerCorrectQuestionSec',0):.1f}s. "
        f"Average time per incorrect question: {ta.get('avgTimePerIncorrectQuestionSec',0):.1f}s.\n"
        f"A significant difference might indicate either rushing on incorrect answers or spending too long on questions you are unsure about."
    )
    
    # Prepare detailed JSON for LLM reference (stringified)
    concept_perf_summary_json = json.dumps([{k: (f"{v:.1f}" if isinstance(v, float) else v) for k,v in c.items()} for c in processed_data.get('conceptPerformance', [])], separators=(',', ':'))
    difficulty_perf_summary_json = json.dumps([{k: (f"{v:.1f}" if isinstance(v, float) else v) for k,v in d.items()} for d in processed_data.get('difficultyPerformance', [])], separators=(',', ':'))
    subjects_detailed_json = json.dumps(processed_data['subjects'], separators=(',', ':'))
    detailed_chapters_json = json.dumps(processed_data['detailed_chapters'], separators=(',', ':'))

    prompt = f"""
    You are an expert academic advisor, 'Insightful Advisor'. Your goal is to generate comprehensive, detailed, and supportive feedback for a student based on their multi-subject assessment data. Analyze the data deeply to find patterns and provide specific, actionable advice.

    Student Name: {student_name}
    Assessment Date: {assessment_date}
    
    OVERALL PERFORMANCE SUMMARY:
    - Overall Accuracy: {overall_accuracy:.1f}% ({correct_answers}/{total_questions} correct)
    - Total Time Taken: {total_time_minutes:.1f} minutes.

    DETAILED PERFORMANCE DATA (FOR YOUR ANALYSIS):
    1. Subject Performance Summary:
    {subjects_text}

    2. Chapter Performance Summary:
    {detailed_chapters_text}

    3. Concept & Difficulty Overview:
       - Concept Strength Summary: {concepts_text}
       - Difficulty Level Performance: {difficulty_text}

    4. Time Management Overview:
       - {time_analysis_text}

    RAW DETAILED DATA (for your in-depth analysis and reference; summarize and interpret, do not just repeat verbatim):
    - Full Subject Data: {subjects_detailed_json}
    - Full Chapter Data: {detailed_chapters_json}
    - Full Concept Performance Data: {concept_perf_summary_json}
    - Full Difficulty Performance Data: {difficulty_perf_summary_json}

    INSTRUCTIONS:
    Your entire response MUST be a single, valid JSON object, wrapped in a markdown code block (```json ... ```).
    The JSON object must have the following exact keys, with string values. Use '\\n' for newlines within strings.

    - "introduction": A personalized, motivating, and detailed introduction (4-6 sentences). Address the student by name. Acknowledge their effort and overall performance, setting a positive, growth-oriented tone.
    - "overall_performance_summary": A concise overview of general performance across all subjects. Mention the overall accuracy and what it generally indicates about their current understanding (e.g., strong foundation, solid understanding with room for improvement, key areas need focus). (3-4 sentences).
    - "subject_breakdown_analysis": Detailed analysis of performance per subject. Highlight strong and weak subjects, using specific accuracy percentages and total questions to support your points. Compare performance between subjects. (Minimum 4-6 sentences, use '\\n' for new paragraphs).
    - "chapter_insights": Specific analysis of performance within different chapters for each subject. Identify chapters of strength and those needing improvement, providing accuracy and time taken. Link chapters to subjects clearly. (Minimum 4-6 sentences, use '\\n' for new paragraphs).
    - "concept_and_difficulty_insights": Go beyond chapters to analyze performance on specific topics/concepts (e.g., "Coulomb's Law," "Molarity"). Identify 2-3 specific concepts where the student is strong and 2-3 that need urgent review, citing accuracy data. Analyze if there's a pattern in difficulty, e.g., struggling with 'Hard' questions across all subjects or only in Physics. (4-6 sentences).
    - "time_accuracy_insights": In-depth analysis of pacing and time management. Compare average time on correct vs. incorrect questions. Offer specific insights, e.g., "You spend significantly less time on incorrect answers in Maths, suggesting you might be making calculation errors while rushing. In contrast, for Physics, the time spent on incorrect answers is high, indicating potential conceptual gaps." (3-5 sentences).
    - "actionable_suggestions": Provide 3-4 highly specific, practical, and personalized suggestions. Link them directly to the weaknesses identified. Instead of "Review weak areas," write "1. Your accuracy in the 'Electrochemistry' chapter was low (e.g., 45%). Dedicate a study session to reviewing 'Faraday's Law' and 'Nernst Equation' concepts, then solve 8-10 practice problems of medium difficulty on this topic."
    - "concluding_remark": A brief, uplifting, and encouraging concluding message (1-2 sentences).

    Example of the required output format:
    ```json
    {{
      "introduction": "Hello {student_name}, let's dive into your recent assessment results...",
      "overall_performance_summary": "Your overall accuracy of {overall_accuracy:.1f}% shows a solid effort...",
      "subject_breakdown_analysis": "In Mathematics, you demonstrated strong performance with...\\nIn contrast, Physics appears to be an area needing more focus...",
      "chapter_insights": "Within Physics, you excelled in 'Capacitance' (80% accuracy). However, 'Electrostatics' (55% accuracy) requires further attention...",
      "concept_and_difficulty_insights": "You have a strong grasp of 'Molarity' (100% accuracy). However, concepts like 'Faraday's Law' proved challenging... Your performance on Easy (90%) and Medium (75%) questions is good, but Hard questions (40%) are a clear area for improvement.",
      "time_accuracy_insights": "Your pacing analysis reveals that you spent an average of 95s on incorrect Physics questions, compared to 60s on correct ones. This suggests you may be struggling with the underlying concepts...",
      "actionable_suggestions": "1. For the 'Electrostatics' chapter, revisit your notes on 'Electric Field' and 'Gauss's Law'. Try creating a concept map to link the ideas.\\n2. To improve on Hard questions, practice breaking them down into smaller, manageable steps before attempting a solution...",
      "concluding_remark": "This report is a tool for your growth. Keep up the dedicated effort!"
    }}
    ```
    """
    return textwrap.dedent(prompt)

def get_llm_feedback(prompt_text, student_name="Student", overall_accuracy=70.0):
    """
    Gets feedback from a Gemini/Gemma LLM by parsing text output, or uses mock data if the API is unavailable.
    This version extracts a JSON object from a markdown code block in the model's text response.
    """
    mock_feedback_dict = {
        "introduction": f"Hello {student_name}! (Mock) Your overall accuracy of {overall_accuracy:.1f}% indicates a solid effort. This comprehensive report aims to highlight your strengths and pinpoint specific areas for growth.",
        "overall_performance_summary": "(Mock) You showed a good grasp of the material generally. Keep building on this foundation.",
        "subject_breakdown_analysis": "(Mock) In Mathematics, you excelled in Calculus. Physics requires more focus, especially in Mechanics.",
        "chapter_insights": "(Mock) Within Physics, your 'Kinematics' chapter showed good progress, while 'Thermodynamics' needs more attention. In Chemistry, 'Organic Chemistry' was a strong area.",
        "concept_and_difficulty_insights": "(Mock) Strong in 'Differentiation' and 'Conduction'. Concepts like 'Newton's Laws' and 'Integration' need review. Hard questions were challenging.",
        "time_accuracy_insights": "(Mock) Your average time per correct question was generally efficient. However, incorrect questions sometimes took longer, suggesting overthinking or conceptual gaps.",
        "actionable_suggestions": "1. (Mock) Dedicate specific study time to the 'Mechanics' subject in Physics, focusing on 'Newton's Laws'.\n2. (Mock) Review 'Integration' concepts and practice varied problem types.\n3. (Mock) For harder questions, try breaking them down into smaller steps or identifying key concepts before attempting a solution.\n4. (Mock) Implement timed practice sessions to improve efficiency without sacrificing accuracy.",
        "concluding_remark": f"(Mock) Keep up the great work, {student_name}! Your dedication to improving will surely lead to even greater success."
    }

    if not GEMINI_AVAILABLE:
        print("Using mock LLM feedback because 'google-generativeai' is not available.")
        return mock_feedback_dict

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY environment variable not set. Using mock LLM feedback.")
        return mock_feedback_dict

    try:
        print("Attempting to call API with model 'gemma-3-27b-it' in text mode...")
        genai.configure(api_key=api_key)
        
        # Reverting to the user-specified model
        model = genai.GenerativeModel(model_name="gemma-3-27b-it")

        # Reverting to text-based generation without forced JSON mode
        generation_config = GenerationConfig(
            temperature=0.7,
        )
        
        response = model.generate_content(
            prompt_text,
            generation_config=generation_config
        )

        print("API call successful. Attempting to extract and parse JSON from text response...")
        
        if response.text:
            text_response = response.text
            # Use regex to find the JSON block within the markdown
            match = re.search(r"```json\s*(\{.*?\})\s*```", text_response, re.DOTALL | re.IGNORECASE)
            
            if match:
                json_string = match.group(1)
                try:
                    feedback_json = json.loads(json_string)
                    expected_keys = ["introduction", "overall_performance_summary", "subject_breakdown_analysis", "chapter_insights", "concept_and_difficulty_insights", "time_accuracy_insights", "actionable_suggestions", "concluding_remark"]
                    if all(key in feedback_json for key in expected_keys):
                        print("Successfully parsed JSON feedback from model's text response.")
                        return feedback_json
                    else:
                        print("Warning: Extracted JSON from model is missing expected keys. Using mock feedback.")
                        return mock_feedback_dict
                except json.JSONDecodeError as je:
                    print(f"Warning: Failed to decode JSON extracted from model's response: {je}. Using mock feedback.")
                    print(f"Problematic JSON string: {json_string[:500]}...")
                    return mock_feedback_dict
            else:
                print("Warning: Could not find a ```json ... ``` block in the model's response. Using mock feedback.")
                print(f"LLM Raw Text: {text_response[:500]}...")
                return mock_feedback_dict
        else:
            print("Warning: LLM response was empty. Using mock feedback.")
            return mock_feedback_dict

    except Exception as e:
        print(f"Error during API call or processing: {e}")
        print("Using mock LLM feedback due to error.")
        return mock_feedback_dict


# --- 3. Chart Generation Functions ---
def generate_bar_chart(labels, values, title, ylabel, filename, color='#4A90E2', rotation=0):
    """Generates a bar chart and saves it as a PNG image."""
    if not labels or not values:
        print(f"Skipping chart '{title}' due to empty data.")
        return None
    plt.figure(figsize=(8, 4))
    bars = plt.bar(labels, values, color=color)
    plt.xlabel("")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=rotation, ha='right' if rotation > 0 else 'center')
    if 'Accuracy' in ylabel:
        plt.ylim(0, 105) # Set Y-axis limit for accuracy charts
    
    plt.tight_layout(pad=2.0)

    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1.5, 
                 f'{yval:.1f}', ha='center', va='bottom', fontsize=8)

    plt.savefig(filename, dpi=300)
    plt.close()
    return filename

def generate_time_comparison_chart(avg_time_correct, avg_time_incorrect, filename):
    """Generates a bar chart comparing average time for correct vs incorrect questions."""
    labels = ['Correct Questions', 'Incorrect Questions']
    values = [avg_time_correct, avg_time_incorrect]
    
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=['#28A745', '#DC3545']) # Green for correct, Red for incorrect
    plt.ylabel('Average Time (Seconds)')
    plt.title('Average Time Spent per Question')
    plt.tight_layout(pad=2.0)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, 
                 f'{yval:.1f}s', ha='center', va='bottom', fontsize=9)

    plt.savefig(filename, dpi=300)
    plt.close()
    return filename


# --- 4. PDF Generation ---
def generate_pdf_report(student_name, assessment_date, subject_list_str, processed_data, llm_feedback, output_pdf_path):
    """Generates a PDF report using ReportLab, including sections for charts."""
    if not processed_data or not llm_feedback:
        print("Cannot generate PDF: Missing processed data or LLM feedback.")
        return

    doc = SimpleDocTemplate(output_pdf_path, pagesize=letter,
                            leftMargin=50, rightMargin=50, topMargin=50, bottomMargin=50)
    story = []

    story.append(Paragraph(f"Student Performance Report", title_style))
    story.append(Spacer(1, 12))
    info_data = [
        ["Student Name:", student_name],
        ["Assessment Date:", assessment_date],
        ["Subjects Assessed:", subject_list_str]
    ]
    info_table = Table(info_data, colWidths=[120, None])
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 11),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('ALIGN', (0,0), (0,-1), 'LEFT'),
        ('ALIGN', (1,0), (1,-1), 'LEFT'),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.darkgrey)
    ]))
    story.append(info_table)
    story.append(Spacer(1, 24))

    story.append(Paragraph("Personalized Introduction", heading1_style))
    story.append(Paragraph(llm_feedback.get("introduction", "N/A").replace('\n', '<br/>'), body_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Overall Performance Snapshot", heading1_style))
    overall_accuracy = processed_data.get('overallAccuracy', 0)
    total_questions = processed_data.get('totalQuestions', 'N/A')
    correct_answers = processed_data.get('correctAnswers', 'N/A')
    total_time_min = processed_data.get('totalAssessmentTimeSeconds', 0) / 60
    overall_text = f"Overall Accuracy: <b>{overall_accuracy:.1f}%</b><br/>Total Questions Attempted: {total_questions}<br/>Correct Answers: {correct_answers}<br/>Total Time Taken: {total_time_min:.1f} minutes"
    story.append(Paragraph(overall_text, body_style))
    story.append(Paragraph(llm_feedback.get("overall_performance_summary", "N/A").replace('\n', '<br/>'), body_style))
    story.append(Spacer(1, 12))

    # --- Subject Performance Section ---
    story.append(Paragraph("Subject-wise Performance Analysis", heading1_style))
    story.append(Paragraph(llm_feedback.get("subject_breakdown_analysis", "N/A").replace('\n', '<br/>'), body_style))
    story.append(Spacer(1, 12))

    if processed_data.get('subjects'):
        subject_names = [s['subjectName'] for s in processed_data['subjects']]
        subject_accuracies = [s['accuracy'] for s in processed_data['subjects']]
        subject_accuracy_chart_path = f"charts/{student_name.replace(' ', '_')}_subject_accuracy_chart.png"
        os.makedirs("charts", exist_ok=True)
        if generate_bar_chart(subject_names, subject_accuracies, "Subject-wise Accuracy", "Accuracy (%)", subject_accuracy_chart_path):
            story.append(Image(subject_accuracy_chart_path, width=400, height=200))
            story.append(Spacer(1, 12))

        story.append(Paragraph("Detailed Subject Performance Data", heading2_style))
        sub_data_table = [["Subject", "Accuracy (%)", "Correct/Total", "Time (min)"]]
        for sub in processed_data['subjects']:
            sub_data_table.append([
                Paragraph(sub['subjectName'], body_style),
                f"{sub['accuracy']:.1f}",
                f"{sub['correctAnswers']}/{sub['totalQuestions']}",
                f"{sub['totalTimeTakenSeconds']/60:.1f}"
            ])
        subject_table = Table(sub_data_table, colWidths=[200,80,100,80])
        subject_table.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#4A90E2")), ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('ALIGN',(0,0),(-1,-1),'CENTER'), ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'), ('FONTSIZE',(0,0),(-1,0),10),
            ('BOTTOMPADDING',(0,0),(-1,0),8), ('BACKGROUND',(0,1),(-1,-1),colors.HexColor("#F0F8FF")),
            ('GRID',(0,0),(-1,-1),1,colors.grey), ('FONTNAME',(0,1),(-1,-1),'Helvetica'),
            ('FONTSIZE',(0,1),(-1,-1),9), ('LEFTPADDING',(0,1),(0,-1),5), ('ALIGN',(0,1),(0,-1),'LEFT')
        ]))
        story.append(subject_table)
    story.append(Spacer(1,12))

    # --- Chapter Performance Section ---
    story.append(Paragraph("Chapter-wise Insights", heading1_style))
    story.append(Paragraph(llm_feedback.get("chapter_insights", "N/A").replace('\n', '<br/>'), body_style))
    story.append(Spacer(1, 12))

    if processed_data.get('detailed_chapters'):
        story.append(Paragraph("Detailed Chapter Performance Data", heading2_style))
        ch_table_data = [["Subject", "Chapter", "Accuracy (%)", "Correct/Total", "Time (min)"]]
        for ch in processed_data['detailed_chapters']:
            ch_table_data.append([
                Paragraph(ch['subjectName'], body_style), Paragraph(ch['name'], body_style),
                f"{ch['accuracy']:.1f}", f"{ch['correctAnswers']}/{ch['totalQuestions']}", f"{ch['timeTakenSeconds']/60:.1f}"
            ])
        chapter_table = Table(ch_table_data, colWidths=[100, 180, 70, 80, 60])
        chapter_table.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#4A90E2")), ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('ALIGN',(0,0),(-1,-1),'CENTER'), ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'), ('FONTSIZE',(0,0),(-1,0),10),
            ('BOTTOMPADDING',(0,0),(-1,0),8), ('BACKGROUND',(0,1),(-1,-1),colors.HexColor("#F0F8FF")),
            ('GRID',(0,0),(-1,-1),1,colors.grey), ('FONTNAME',(0,1),(-1,-1),'Helvetica'),
            ('FONTSIZE',(0,1),(-1,-1),9), ('LEFTPADDING',(0,1),(1,-1),5), ('ALIGN',(0,1),(1,-1),'LEFT')
        ]))
        story.append(chapter_table)
    story.append(Spacer(1,12))

    # --- Concept and Difficulty Section ---
    story.append(Paragraph("Concept and Difficulty Insights", heading1_style))
    story.append(Paragraph(llm_feedback.get("concept_and_difficulty_insights", "N/A").replace('\n', '<br/>'), body_style))
    story.append(Spacer(1,12))
    
    charts_in_flow = []
       

    # Difficulty Performance Chart
    if processed_data.get('difficultyPerformance'):
        levels = [d['level'] for d in processed_data['difficultyPerformance']]
        accuracies = [d['accuracy'] for d in processed_data['difficultyPerformance']]
        difficulty_chart_path = f"charts/{student_name.replace(' ', '_')}_difficulty_accuracy_chart.png"
        if generate_bar_chart(levels, accuracies, "Accuracy by Difficulty Level", "Accuracy (%)", difficulty_chart_path):
            charts_in_flow.append(Image(difficulty_chart_path, width=400, height=200))
    
    if charts_in_flow:
        story.append(Table([charts_in_flow], colWidths=[doc.width/len(charts_in_flow)]*len(charts_in_flow)))
        story.append(Spacer(1,12))


    # --- Time Management Section ---
    story.append(Paragraph("Time Management and Accuracy Insights", heading1_style))
    story.append(Paragraph(llm_feedback.get("time_accuracy_insights", "N/A").replace('\n', '<br/>'), body_style))
    story.append(Spacer(1,12))

    ta = processed_data.get('timeAnalysis', {})
    if ta.get('numCorrect', 0) > 0 or ta.get('numIncorrect', 0) > 0:
        avg_time_correct = ta.get('avgTimePerCorrectQuestionSec', 0)
        avg_time_incorrect = ta.get('avgTimePerIncorrectQuestionSec', 0)
        time_chart_path = f"charts/{student_name.replace(' ', '_')}_time_comparison_chart.png"
        if generate_time_comparison_chart(avg_time_correct, avg_time_incorrect, time_chart_path):
            story.append(Image(time_chart_path, width=350, height=200))
            story.append(Spacer(1,12))

    # --- Suggestions and Conclusion ---
    story.append(Paragraph("Actionable Suggestions", heading1_style))
    suggestions_text = llm_feedback.get("actionable_suggestions", "N/A")
    for part in suggestions_text.split('\n'):
        part_stripped = part.strip()
        if not part_stripped: continue
        
        match = re.match(r'^(\d+\.)\s*(.*)', part_stripped)
        if match:
            bullet_char = match.group(1)
            text_content = match.group(2)
            story.append(Paragraph(text_content, bullet_style, bulletText=bullet_char))
        else:
            story.append(Paragraph(part_stripped, body_style))
    story.append(Spacer(1,12))

    story.append(Paragraph("Concluding Remark", heading1_style))
    story.append(Paragraph(llm_feedback.get("concluding_remark", "N/A").replace('\n', '<br/>'), body_style))
    story.append(Spacer(1,24))
    story.append(Paragraph(f"Report Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by Insightful Advisor",ParagraphStyle('Footer',fontSize=8,alignment=TA_CENTER,textColor=colors.grey)))

    try:
        doc.build(story)
        print(f"Successfully generated PDF report: {output_pdf_path}")
    except Exception as e:
        print(f"Error generating PDF: {e}")
    
    # Clean up generated chart images
    charts_dir = "charts"
    if os.path.exists(charts_dir):
        for f in os.listdir(charts_dir):
            if f.startswith(student_name.replace(' ', '_')) and f.endswith(".png"):
                try:
                    os.remove(os.path.join(charts_dir, f))
                except OSError as e:
                    print(f"Error removing chart file {f}: {e}")

# --- Main Execution ---
def main(json_file_path, output_pdf_name_template="Student_Performance_Report_{student_name_safe}.pdf"):
    print("Starting Student Performance Feedback System...")
    print("Ensure 'google-generativeai' is installed ('pip install google-generativeai') and the GEMINI_API_KEY environment variable is set in a .env file.")
    raw_data = load_data(json_file_path)
    if not raw_data:
        print("Exiting due to data loading issues."); return

    submissions = raw_data if isinstance(raw_data, list) else [raw_data]
    print(f"Found {len(submissions)} submission(s). Generating a report for each.")

    if not submissions:
        print("No valid submissions found to process. Exiting."); return

    for i, submission_data in enumerate(submissions):
        print(f"\n--- Processing submission {i+1}/{len(submissions)} ---")
        
        processed_data = process_data(dict(submission_data))

        if not processed_data or processed_data.get('totalQuestions', 0) == 0:
            print(f"Skipping submission {i+1} due to no processable question data found."); continue
        
        student_name_raw = processed_data.get('studentName', f'Student_{i+1}')
        student_name_safe = re.sub(r'[^\w\-_\. ]', '', student_name_raw).replace(" ", "_")
        
        current_output_pdf_name = output_pdf_name_template.format(student_name_safe=student_name_safe, index=i+1)

        llm_prompt = format_data_for_llm(processed_data)
        llm_responses = get_llm_feedback(llm_prompt, student_name=processed_data['studentName'], overall_accuracy=processed_data['overallAccuracy'])
        
        generate_pdf_report(
            student_name=processed_data['studentName'],
            assessment_date=processed_data['assessmentDate'],
            subject_list_str=processed_data['subject_list_str'],
            processed_data=processed_data,
            llm_feedback=llm_responses,
            output_pdf_path=current_output_pdf_name
        )
    print("\n--- All submissions processed. ---")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("\nUsage: python feedback_system.py <path_to_json_file> [output_pdf_name_template.pdf]")
        print("Example: python feedback_system.py sample_submission_analysis_1.json \"Report_{student_name_safe}.pdf\"")
        print("\nCreating a dummy 'dummy_submission.json' for testing as no file was provided...")
        
        # Dummy data matching the 'sample_submission_analysis_1.json' structure
        dummy_data = [
            {
                "_id": {"$oid": "683bf660e0086564267a7278"},
                "studentName": "Alice Wonderland",
                "assessmentDate": "2024-05-20",
                "test": {
                    "syllabus": "<h1>QPT 1 Syllabus</h1><h2>Physics</h2><h2>Chemistry</h2><h2>Maths</h2>",
                },
                "sections": [
                    {
                        "sectionId": {"title": "Physics Section"},
                        "questions": [
                            {"questionId": {"level": "easy", "chapters": [{"title": "Electrostatics"}], "topics": [{"title": "Coulomb's Law"}]}, "status": "answered", "markedOptions": [{"isCorrect": True}], "timeTaken": 60},
                            {"questionId": {"level": "medium", "chapters": [{"title": "Electrostatics"}], "topics": [{"title": "Electric Field"}]}, "status": "answered", "markedOptions": [{"isCorrect": False}], "timeTaken": 90},
                            {"questionId": {"level": "hard", "chapters": [{"title": "Capacitance"}], "topics": [{"title": "Parallel Plate"}]}, "status": "answered", "markedOptions": [{"isCorrect": True}], "timeTaken": 120},
                            {"questionId": {"level": "medium", "chapters": [{"title": "Capacitance"}], "topics": [{"title": "Dielectrics"}]}, "status": "markedReview", "markedOptions": [{"isCorrect": False}], "timeTaken": 100}
                        ]
                    },
                    {
                        "sectionId": {"title": "Chemistry Section"},
                        "questions": [
                            {"questionId": {"level": "easy", "chapters": [{"title": "Solutions"}], "topics": [{"title": "Concentration"}]}, "status": "answered", "inputValue": {"isCorrect": True}, "timeTaken": 50},
                            {"questionId": {"level": "medium", "chapters": [{"title": "Electrochemistry"}], "topics": [{"title": "Electrolysis"}]}, "status": "answered", "inputValue": {"isCorrect": True}, "timeTaken": 80},
                            {"questionId": {"level": "medium", "chapters": [{"title": "Electrochemistry"}], "topics": [{"title": "Redox Reactions"}]}, "status": "notAnswered", "inputValue": {"isCorrect": False}, "timeTaken": 110}
                        ]
                    },
                    {
                        "sectionId": {"title": "Maths Section"},
                        "questions": [
                            {"questionId": {"level": "hard", "chapters": [{"title": "Sets and Relations"}], "topics": [{"title": "Set Operations"}]}, "status": "answered", "markedOptions": [{"isCorrect": True}], "timeTaken": 95},
                            {"questionId": {"level": "easy", "chapters": [{"title": "Functions"}], "topics": [{"title": "Domain and Range"}]}, "status": "markedReview", "markedOptions": [{"isCorrect": False}], "timeTaken": 40}
                        ]
                    }
                ]
            }
        ]
        
        os.makedirs('charts', exist_ok=True)
        with open('dummy_submission.json', 'w', encoding='utf-8') as f:
            json.dump(dummy_data, f, indent=2)
            
        json_file = 'dummy_submission.json'
        output_pdf_template = "Report_{student_name_safe}.pdf"
        print(f"Running with dummy data: '{json_file}', outputting to '{output_pdf_template}'\n")
        main(json_file, output_pdf_template)
    else:
        json_file = sys.argv[1]
        output_pdf_template = "Student_Performance_Report_{student_name_safe}.pdf"
        if len(sys.argv) > 2:
            output_pdf_template = sys.argv[2]
        
        os.makedirs('charts', exist_ok=True)
        main(json_file, output_pdf_template)