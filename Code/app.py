import os
import tempfile
import shutil
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, send_file
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, date
import re
from io import BytesIO

# Imports for PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# Imports for Email Sending
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# LangChain + Chroma + LegalBERT
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings

# Added for the new advocate search feature
import json

# Disable ONNX embeddings in Chroma to avoid a dependency conflict
os.environ["CHROMADB_DEFAULT_EMBEDDING_FUNCTION"] = "null"

load_dotenv()

# ==================== Flask App Setup ====================
app = Flask(__name__)

# User authentication setup
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a-very-secret-key-that-you-should-change')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///user.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# RAG setup
UPLOAD_FOLDER = 'data'
#DB_PATH = 'chroma_db'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#os.makedirs(DB_PATH, exist_ok=True)

# Email configuration
# Make sure to set these in your .env file
# GMAIL_EMAIL=your_email@gmail.com
# GMAIL_APP_PASSWORD=your_app_password
EMAIL_USER = os.environ.get('GMAIL_EMAIL')
EMAIL_PASSWORD = os.environ.get('GMAIL_APP_PASSWORD')
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 465 # For SSL

# ==================== Database Model ====================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    address = db.Column(db.String(255), nullable=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'

class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    severity = db.Column(db.String(500))
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    query = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# ==================== RAG Functions ====================
rag_chain = None
vectorstore = None
case_severity = None
llm = Ollama(model="llama3.2")
# Use Ollama embeddings for simplicity
embedding_model = OllamaEmbeddings(model="llama3.2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def analyze_case_severity(document_text):
    """Analyze case severity using LLM"""
    severity_prompt = f"""
    Analyze the following legal document and determine the case severity level.
    Consider factors like: potential penalties, financial impact, criminal vs civil nature, urgency, complexity.
    
    Respond with ONLY one of these severity levels and a brief reason:
    - LOW: Minor civil matters, small claims, routine contracts
    - MEDIUM: Significant civil disputes, employment issues, moderate financial impact
    - HIGH: Criminal charges, major financial disputes, urgent injunctions
    - CRITICAL: Felony charges, bankruptcy, major corporate litigation
    
    Document: {document_text[:2000]}...
    
    Format: SEVERITY_LEVEL: Reason
    """
    
    try:
        llm = Ollama(model="llama3.2")
        response = llm.invoke(severity_prompt)
        return response.strip()
    except Exception as e:
        return f"Unable to analyze severity: {str(e)}"

def process_document_and_ingest_to_db(file_path):
    global rag_chain, vectorstore
    try:
        # Clear existing data
        #if os.path.exists(DB_PATH):
            #shutil.rmtree(DB_PATH)
        
        # Load the document
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            return False # Unsupported file type

        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        
        # Analyze case severity
        full_text = "\n".join([doc.page_content for doc in documents])
        severity_analysis = analyze_case_severity(full_text)

        # Ingest into ChromaDB with temporary directory
        vectorstore = Chroma.from_documents(
            documents=docs, 
            embedding=embedding_model,
            persist_directory=tempfile.mkdtemp()
        )

        # Set up the RAG chain
        rag_prompt_template = """
        You are a legal AI assistant. Your purpose is to provide legal information based on the context provided.
        Answer the question as concisely as possible based only on the following context.
        If you cannot find the answer in the context, do not make up an answer. Simply state that you cannot find the answer in the provided document.

        Context: {context}

        Question: {question}
        """
        rag_prompt = PromptTemplate.from_template(rag_prompt_template)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        rag_chain = (
            {"context": retriever | format_docs,"question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        
        # Store severity analysis globally (in production, store in database)
        global case_severity
        case_severity = severity_analysis
        
        return True
    except Exception as e:
        print(f"Error processing document: {e}")
        return False

# ==================== Search Advocates Function ====================
def get_case_type_from_document(document_text):
    """Extract case type from document using LLM"""
    prompt = f"""
    Analyze this legal document and identify the primary case type.
    Respond with ONE of: Criminal, Civil, Corporate, Family Law, Employment, Real Estate, Commercial Law, Contracts
    
    Document: {document_text[:1500]}...
    
    Case Type:
    """
    try:
        llm = Ollama(model="llama3.2")
        return llm.invoke(prompt).strip()
    except:
        return "General"

def search_advocates_data(location, query):
    """
    Simulates searching for advocates and returns a structured list.
    """
    search_results = [
        {"name": "Eman Rahim", "experience": "34 Years", "areas": ["Contracts", "Criminal", "Civil"], "phone": "+91 95854*****"},
        {"name": "Rama Subramanian Ammamuthu", "experience": "20 Years", "areas": ["real estate", "Commercial Law"], "phone": "+91 93456*****"},
        {"name": "Adv. Deepak Chandrakanth", "experience": "N/A", "areas": ["General Services", "Family Law"], "phone": "N/A"},
        {"name": "S.SHANMUGAM", "experience": "N/A", "areas": ["Corporate Law", "Commercial Law"], "phone": "+91 97876 75250", "email": "s.shanmugamadv84@gmail.com"},
        {"name": "S.MOHAMED YUNUS", "experience": "N/A", "areas": ["General Services", "Commercial Law"], "phone": "+91 88072 60676", "email": "smyunus.adv@gmail.com"},
        {"name": "K.SUSEELA", "experience": "N/A", "areas": ["Civil", "Family Law"], "phone": "+91 94455 77779", "email": "s.suseelaadv90@gmail.com"},
        {"name": "Advocate Nithiyanandan", "experience": "N/A", "areas": ["Corporate Law"], "phone": "N/A"},
        {"name": "Advocate A Prabu Armugam", "experience": "N/A", "areas": ["Corporate Law"], "phone": "N/A"}
    ]
    return search_results

# ==================== Routes ====================

@app.route('/')
def welcome():
    # Renders the new stylish landing page.
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        address = request.form['address']

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists. Please choose a different one.')
            return redirect(url_for('register'))

        new_user = User(username=username, email=email, address=address)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/home')
def home():
    if 'username' in session:
        return render_template('home.html', username=session['username'])
    return redirect(url_for('login'))



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/features')
def features():
    return render_template('features.html')




@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('welcome'))

@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        return render_template('dashboard.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/assistant')
def assistant():
    if 'username' in session:
        return render_template('assistant.html')
    return redirect(url_for('login'))

@app.route('/upload', methods=['POST'])
def upload_document():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    try:
        file.save(file_path)
        success = process_document_and_ingest_to_db(file_path)
        if success:
            global case_severity
            # Store in database
            doc = Document(
                user_id=session['user_id'],
                filename=file.filename,
                file_path=file_path,
                severity=case_severity
            )
            db.session.add(doc)
            db.session.commit()
            
            return jsonify({
                'message': '✅ Document uploaded & processed successfully!',
                'severity': case_severity
            }), 200
        else:
            return jsonify({'error': 'Failed to process document. Use PDF or DOCX'}), 500
    except Exception as e:
        return jsonify({'error': f"Failed to save file: {str(e)}"}), 500

@app.route('/chat', methods=['POST'])
def chat_with_ai():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    global rag_chain, case_severity
    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    if rag_chain is None:
        return jsonify({'response': '⚠️ Please upload a legal document first.'}), 200

    # Handle severity queries specially
    if 'severity' in query.lower() and 'case' in query.lower():
        if case_severity:
            return jsonify({'response': f"📊 Case Severity Analysis: {case_severity}"}), 200
        else:
            return jsonify({'response': '⚠️ No severity analysis available. Please upload a document first.'}), 200
    
    # Handle lawyer recommendation queries
    if any(word in query.lower() for word in ['lawyer', 'advocate', 'attorney', 'recommend', 'suggest']):
        try:
            doc = Document.query.filter_by(user_id=session['user_id']).order_by(Document.uploaded_at.desc()).first()
            if not doc:
                return jsonify({'response': '⚠️ Please upload a document first to get lawyer recommendations.'}), 200
            
            # Load document
            if doc.file_path.endswith('.pdf'):
                loader = PyPDFLoader(doc.file_path)
            else:
                loader = Docx2txtLoader(doc.file_path)
            
            documents = loader.load()
            full_text = "\n".join([d.page_content for d in documents])
            case_type = get_case_type_from_document(full_text)
            
            # Get matching advocates
            all_advocates = search_advocates_data("", "")
            recommended = []
            for adv in all_advocates:
                for area in adv['areas']:
                    if case_type.lower() in area.lower() or area.lower() in case_type.lower():
                        recommended.append(adv)
                        break
            
            if not recommended:
                recommended = all_advocates[:3]
            
            # Format response
            response = f"📋 **Case Type:** {case_type}\n"
            response += f"⚖️ **Severity:** {case_severity}\n\n"
            response += "👨‍⚖️ **Recommended Lawyers:**\n\n"
            for i, lawyer in enumerate(recommended[:5], 1):
                response += f"{i}. **{lawyer['name']}**\n"
                response += f"   Experience: {lawyer['experience']}\n"
                response += f"   Specialization: {', '.join(lawyer['areas'])}\n"
                response += f"   Phone: {lawyer['phone']}\n"
                if 'email' in lawyer:
                    response += f"   Email: {lawyer['email']}\n"
                response += "\n"
            
            return jsonify({'response': response}), 200
        except Exception as e:
            return jsonify({'response': f'⚠️ Error getting recommendations: {str(e)}'}), 200

    try:
        response = rag_chain.invoke(query)

        print(f"💬 Query: {query}")
        print(f"🤖 Response: {response}")

        if not response or response.strip() == "":
            response = "⚠️ No answer generated. Check if the LLM is running."
        
        # Store chat history
        chat = ChatHistory(
            user_id=session['user_id'],
            query=query,
            response=response
        )
        db.session.add(chat)
        db.session.commit()

        return jsonify({'response': response}), 200
    except Exception as e:
        print(f"❌ Chat error: {e}")
        return jsonify({'response': f"❌ An error occurred: {str(e)}"}), 500

@app.route('/extract_clause', methods=['POST'])
def extract_clause():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    global rag_chain
    data = request.json
    clause_type = data.get('clause_type')

    if not clause_type:
        return jsonify({'response': '❌ Please provide a clause type to extract.'}), 400

    if rag_chain is None:
        return jsonify({'response': '⚠️ Please upload a legal document first.'}), 200
    
    try:
        # Construct a specific query for clause extraction
        extract_query = f"Extract the full text of the '{clause_type}' clause from the provided document. If multiple clauses exist, extract all of them. If the clause is not found, state that."
        
        response = rag_chain.invoke(extract_query)
        
        if not response or response.strip() == "":
            response = f"⚠️ The '{clause_type}' clause was not found in the document or the LLM failed to generate a response."
            
        return jsonify({'response': response}), 200
    
    except Exception as e:
        print(f"❌ Extraction error: {e}")
        return jsonify({'response': f"❌ Extraction error: {e}"}), 500

@app.route('/generate_document', methods=['POST'])
def generate_document():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.json
    doc_type = data.get('docType')
    
    if not doc_type:
        return jsonify({'document': '❌ Please provide the document type.'}), 400

    # Define a prompt template for generating a legal document
    prompt_template = """
    You are a legal document drafting AI. Your task is to provide a standardized template or boilerplate for a legal document.
    The template should be clearly structured with placeholders for variable information like names and addresses.
    Do not fill in the personal details provided by the user.

    Please provide a legal boilerplate template for a **{doc_type}**.
    Include the standard sections and clauses that would be found in such a document, using placeholders like [PARTY A NAME], [PARTY A ADDRESS], etc.
    """
    
    formatted_prompt = prompt_template.format(doc_type=doc_type)
    
    # Use the existing LLM to generate the document
    try:
        llm = Ollama(model="llama3.2")
        response = llm.invoke(formatted_prompt)

        # Add a note to the response
        final_document = f"**{doc_type} Boilerplate Template**\n\n{response}\n\n"
        final_document += "---"
        final_document += "\n\nDisclaimer: This is an AI-generated template. It is recommended to consult with a legal professional before using this document. You will need to manually replace the bracketed placeholders with the correct information."

        return jsonify({'document': final_document}), 200

    except Exception as e:
        print(f"❌ Document generation error: {e}")
        return jsonify({'document': f"❌ Failed to generate document: {e}"}), 500

@app.route('/download_pdf')
def download_pdf():
    doc_type = request.args.get('doc_type', 'Legal Document')
    content = request.args.get('content', 'No content available.')
    
    buffer = BytesIO()
    
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    flowables = []
    flowables.append(Paragraph(f"<b>{doc_type}</b>", styles['Title']))
    flowables.append(Paragraph("<br/>", styles['Normal']))
    
    for line in content.split('\n'):
        flowables.append(Paragraph(line, styles['Normal']))
    
    doc.build(flowables)
    
    buffer.seek(0)
    
    return send_file(buffer, as_attachment=True, download_name=f"{doc_type.replace(' ', '_')}.pdf", mimetype='application/pdf')

@app.route('/recommend_lawyers', methods=['POST'])
def recommend_lawyers():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    global rag_chain, case_severity
    
    if rag_chain is None:
        return jsonify({'error': 'Please upload a document first'}), 400
    
    try:
        # Get document content
        doc = Document.query.filter_by(user_id=session['user_id']).order_by(Document.uploaded_at.desc()).first()
        if not doc:
            return jsonify({'error': 'No document found'}), 400
        
        # Load document to extract case type
        if doc.file_path.endswith('.pdf'):
            loader = PyPDFLoader(doc.file_path)
        else:
            loader = Docx2txtLoader(doc.file_path)
        
        documents = loader.load()
        full_text = "\n".join([d.page_content for d in documents])
        case_type = get_case_type_from_document(full_text)
        
        # Get all advocates
        all_advocates = search_advocates_data("", "")
        
        # Filter advocates by case type
        recommended = []
        for adv in all_advocates:
            for area in adv['areas']:
                if case_type.lower() in area.lower() or area.lower() in case_type.lower():
                    recommended.append(adv)
                    break
        
        # If no match, return all
        if not recommended:
            recommended = all_advocates[:3]
        
        return jsonify({
            'case_type': case_type,
            'severity': case_severity,
            'recommended_lawyers': recommended
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to recommend lawyers: {str(e)}'}), 500

@app.route('/find_advocates', methods=['POST'])
def find_advocates():
    data = request.json
    location = data.get('location')
    specialty = data.get('specialty')

    if not location:
        return jsonify({'error': 'Please provide a location.'}), 400
    
    query = f"advocates in {location}"
    if specialty:
        query = f"{specialty} {query}"

    advocates = search_advocates_data(location, query)
    
    return jsonify({'advocates': advocates}), 200

@app.route('/draft_legal_mail', methods=['POST'])
def draft_legal_mail():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.json
    issue = data.get('issue')
    sender_name = data.get('sender_name')
    sender_address = data.get('sender_address')
    
    if not issue or not sender_name or not sender_address:
        return jsonify({'error': 'Please provide all required information.'}), 400

    drafting_prompt = f"""
    You are a professional business correspondent. Draft a formal and strongly-worded letter of complaint based on the following issue.
    The letter should begin with the sender's details. It should clearly state the problem, the required actions from the recipient, and a reasonable deadline for a response.

    Sender's Name: {sender_name}
    Sender's Address: {sender_address}

    Problem: {issue}
    """
    
    try:
        llm = Ollama(model="llama3.2")
        letter_body = llm.invoke(drafting_prompt)
        
        disclaimer = "\n\n---"
        disclaimer += "\n\nDisclaimer: This is an AI-generated draft. It is for informational purposes only and does not constitute legal advice. Please review the content carefully and consult a legal professional before sending."
        
        final_letter = letter_body + disclaimer
        
        return jsonify({'letter_content': final_letter}), 200

    except Exception as e:
        print(f"❌ Drafting error: {e}")
        return jsonify({'error': f'Failed to draft letter: {str(e)}'}), 500

@app.route('/admin/dashboard')
def admin_dashboard():
    return render_template('admin_dashboard.html')

@app.route('/admin/data')
def admin_data():
    users = User.query.all()
    documents = Document.query.all()
    chats = ChatHistory.query.order_by(ChatHistory.created_at.desc()).limit(100).all()
    
    return jsonify({
        'users': [{'id': u.id, 'username': u.username, 'email': u.email, 'address': u.address} for u in users],
        'documents': [{'id': d.id, 'user_id': d.user_id, 'filename': d.filename, 'severity': d.severity, 'uploaded_at': d.uploaded_at.isoformat()} for d in documents],
        'chats': [{'id': c.id, 'user_id': c.user_id, 'query': c.query, 'response': c.response, 'created_at': c.created_at.isoformat()} for c in chats]
    }), 200

@app.route('/get_chat_history', methods=['GET'])
def get_chat_history():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    chats = ChatHistory.query.filter_by(user_id=session['user_id']).order_by(ChatHistory.created_at.desc()).limit(50).all()
    history = [{'query': c.query, 'response': c.response, 'time': c.created_at.isoformat()} for c in chats]
    return jsonify({'history': history}), 200

@app.route('/get_documents', methods=['GET'])
def get_documents():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    docs = Document.query.filter_by(user_id=session['user_id']).order_by(Document.uploaded_at.desc()).all()
    documents = [{'id': d.id, 'filename': d.filename, 'severity': d.severity, 'uploaded_at': d.uploaded_at.isoformat()} for d in docs]
    return jsonify({'documents': documents}), 200

@app.route('/get_severity', methods=['GET'])
def get_case_severity():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    global case_severity
    if case_severity is None:
        return jsonify({'severity': 'No document analyzed yet'}), 200
    
    return jsonify({'severity': case_severity}), 200

@app.route('/translate', methods=['POST'])
def translate_text():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.json
    text = data.get('text')
    target_lang = data.get('target_language', 'en')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    translation_prompt = f"""
    Translate the following legal text to {target_lang}. Maintain legal terminology accuracy and formal tone.
    
    Text: {text}
    
    Provide only the translation without explanations.
    """
    
    try:
        llm = Ollama(model="llama3.2")
        translated = llm.invoke(translation_prompt)
        return jsonify({'translated_text': translated.strip()}), 200
    except Exception as e:
        return jsonify({'error': f'Translation failed: {str(e)}'}), 500

@app.route('/simplify_legal_text', methods=['POST'])
def simplify_legal_text():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.json
    text = data.get('text')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    simplify_prompt = f"""
    Simplify the following legal text into plain language that a non-lawyer can understand.
    Explain complex legal terms in simple words.
    
    Legal Text: {text}
    
    Simplified Explanation:
    """
    
    try:
        llm = Ollama(model="llama3.2")
        simplified = llm.invoke(simplify_prompt)
        return jsonify({'simplified_text': simplified.strip()}), 200
    except Exception as e:
        return jsonify({'error': f'Simplification failed: {str(e)}'}), 500

@app.route('/voice_to_text', methods=['POST'])
def voice_to_text():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    
    audio_file = request.files['audio']
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"voice_{session['user_id']}_temp.webm")
    wav_path = os.path.join(app.config['UPLOAD_FOLDER'], f"voice_{session['user_id']}.wav")
    
    try:
        audio_file.save(temp_path)
        
        # Try to convert using pydub (requires ffmpeg)
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(temp_path)
            audio.export(wav_path, format="wav")
        except:
            # Fallback: try direct recognition on webm
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
            os.remove(temp_path)
            return jsonify({'text': text}), 200
        
        # Use speech recognition on converted WAV
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        
        os.remove(temp_path)
        os.remove(wav_path)
        return jsonify({'text': text}), 200
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)
        return jsonify({'error': f'Speech recognition failed: {str(e)}'}), 500

@app.route('/text_to_speech', methods=['POST'])
def text_to_speech():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.json
    text = data.get('text')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"tts_{session['user_id']}.mp3")
    
    try:
        from gtts import gTTS
        
        # Remove old file if exists
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(audio_path)
        
        return send_file(audio_path, mimetype='audio/mpeg', as_attachment=False, download_name='response.mp3')
    except Exception as e:
        return jsonify({'error': f'Text-to-speech failed: {str(e)}'}), 500

@app.route('/send_legal_mail', methods=['POST'])
def send_legal_mail():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.json
    recipient_email = data.get('recipient_email')
    letter_content = data.get('letter_content')
    
    if not recipient_email or not letter_content:
        return jsonify({'error': 'Email content or recipient is missing.'}), 400

    if not EMAIL_USER or not EMAIL_PASSWORD:
        return jsonify({'error': 'Email credentials not configured on the server.'}), 500
    
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = recipient_email
        msg['Subject'] = "URGENT ATTENTION: Formal Complaint Regarding a Problem"
        
        msg.attach(MIMEText(letter_content, 'plain'))
        
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_USER, recipient_email, msg.as_string())
        
        return jsonify({'message': '✅ Your formal letter has been sent successfully!'}), 200

    except Exception as e:
        print(f"❌ Email sending error: {e}")
        return jsonify({'error': f'Failed to send email: {str(e)}'}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)