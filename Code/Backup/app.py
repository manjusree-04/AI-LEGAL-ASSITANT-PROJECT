import os
import shutil
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import re
import json

# LangChain + Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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
DB_PATH = 'chroma_db'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DB_PATH, exist_ok=True)

# Global variables for RAG chain
rag_chain = None
vectorstore = None

# ==================== Database Models ====================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String(255), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

# ==================== RAG Functions ====================
def initialize_rag_chain():
    """Initializes the RAG chain and vector store."""
    global rag_chain, vectorstore
    
    # Check if a document is already uploaded and vector store exists
    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        try:
            embeddings = OllamaEmbeddings(model="llama3.2")
            vectorstore = Chroma(
                persist_directory=DB_PATH,
                embedding_function=embeddings
            )
            
            # Setup the RAG chain
            llm = Ollama(model="llama3.2")
            retriever = vectorstore.as_retriever()
            
            prompt_template = PromptTemplate(
                template="""
                You are a legal assistant. Answer the question based on the following context. 
                If the question cannot be answered from the provided context, say "I can't assist with that."
                
                Context: {context}
                
                Question: {question}
                """,
                input_variables=["context", "question"],
            )
            
            rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt_template
                | llm
                | StrOutputParser()
            )
            print("✅ RAG chain initialized successfully.")
            return True
        
        except Exception as e:
            print(f"❌ Error initializing RAG chain: {e}")
            rag_chain = None
            vectorstore = None
            return False
    else:
        print("⚠️ No documents uploaded. RAG chain is not active.")
        rag_chain = None
        vectorstore = None
        return False

# ==================== Simulated Advocate Data ====================
def search_advocates_data(location, query):
    """Simulates a search for legal advocates."""
    # This is a mock function. In a real app, you'd use a service like Google Maps API.
    # The output is a hardcoded response based on the search I did.
    mock_data = {
        "Chennai": [
            {
                "name": "S.P. Law Associates",
                "specialty": "Civil Law",
                "phone": "+91 98765 43210",
                "address": "123, High Court Road, Chennai",
                "website": "http://www.splawassociates.com"
            },
            {
                "name": "K.R. & Partners",
                "specialty": "Corporate Law",
                "phone": "+91 99887 76655",
                "address": "45, Mount Road, Chennai",
                "website": "http://www.krpartners.in"
            }
        ],
        "Mumbai": [
            {
                "name": "Legal Nexus Firm",
                "specialty": "Criminal Law",
                "phone": "+91 87654 32109",
                "address": "789, Fort Street, Mumbai",
                "website": "http://www.legalnexusfirm.com"
            },
            {
                "name": "Bhatia & Co.",
                "specialty": "Family Law",
                "phone": "+91 88776 65544",
                "address": "101, Marine Drive, Mumbai",
                "website": "http://www.bhatiandco.in"
            }
        ]
    }
    
    city = location.strip().title()
    specialty_match = re.search(r'(corporate|civil|criminal|family) law', query, re.I)
    
    if city in mock_data:
        results = mock_data[city]
        if specialty_match:
            specialty = specialty_match.group(1).title() + " Law"
            filtered_results = [advocate for advocate in results if advocate['specialty'] == specialty]
            return filtered_results if filtered_results else results
        return results
    
    return []

# ==================== Application Routes ====================

@app.before_request
def create_tables():
    """Create database tables before the first request."""
    if not os.path.exists('user.db'):
        with app.app_context():
            db.create_all()

@app.route('/')
def index():
    if 'user_id' in session:
        return render_template('index.html')
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if not username or not password:
            flash('Username and password are required.')
            return redirect(url_for('register'))
        
        user = User.query.filter_by(username=username).first()
        if user:
            flash('Username already exists. Please choose a different one.')
            return redirect(url_for('register'))
        
        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            flash('Logged in successfully!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/upload_document', methods=['POST'])
def upload_document():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    file_extension = file.filename.rsplit('.', 1)[1].lower()
    
    if file_extension not in ['pdf', 'docx']:
        return jsonify({'error': 'Unsupported file type. Please upload a .pdf or .docx'}), 400
    
    # Clear previous Chroma DB and data folder
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    try:
        if file_extension == 'pdf':
            loader = PyPDFLoader(filepath)
        elif file_extension == 'docx':
            loader = Docx2txtLoader(filepath)
        
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        
        chunks = text_splitter.split_documents(documents)
        
        embeddings = OllamaEmbeddings(model="llama3.2")
        
        global vectorstore
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_PATH
        )
        
        # Persist the vector store
        vectorstore.persist()
        
        # Initialize the RAG chain after successful upload and indexing
        initialize_rag_chain()
        
        # Store document info in the database
        user_id = session['user_id']
        new_doc = Document(file_name=file.filename, user_id=user_id)
        db.session.add(new_doc)
        db.session.commit()
        
        return jsonify({'message': f'Document "{file.filename}" uploaded and processed successfully!'}), 200
        
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return jsonify({'error': f'Failed to process document: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.json
    query = data.get('query')
    
    if not query:
        return jsonify({'response': 'No query provided'}), 400
    
    if rag_chain is None:
        return jsonify({'response': '⚠️ Please upload a legal document first.'}), 200
    
    try:
        response = rag_chain.invoke(query)
        
        print(f"💬 Query: {query}")
        print(f"🤖 Response: {response}")
        
        if not response or response.strip() == "":
            response = "⚠️ No answer generated. Check if the LLM is running."
            
        return jsonify({'response': response}), 200
    except Exception as e:
        print(f"❌ Chat error: {e}")
        return jsonify({'response': f"❌ An error occurred: {str(e)}"}), 500

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

    # REVISED PROMPT: Use more general, business-oriented language
    drafting_prompt = f"""
    You are a professional business correspondent. Draft a formal letter to a person or business to address a serious issue. The letter should clearly state the problem and demand action to rectify it. The letter must address an overdue financial matter and include a specific deadline for a response.

    Sender's Details:
    Name: {sender_name}
    Address: {sender_address}

    Problem: {issue}
    """
    
    try:
        llm = Ollama(model="llama3.2")
        letter_body = llm.invoke(drafting_prompt)
        
        # Add a clear disclaimer to the generated content
        disclaimer = "\n\n---"
        disclaimer += "\n\nDisclaimer: This is an AI-generated draft. It is for informational purposes only and does not constitute legal advice. Please review the content carefully and consult a legal professional before sending."
        
        final_letter = letter_body + disclaimer
        
        return jsonify({'letter_content': final_letter}), 200

    except Exception as e:
        print(f"❌ Drafting error: {e}")
        return jsonify({'error': f'Failed to draft letter: {str(e)}'}), 500

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

@app.route('/find_advocates', methods=['POST'])
def find_advocates():
    """Backend route to find nearby advocates."""
    data = request.json
    location = data.get('location')
    specialty = data.get('specialty')

    if not location:
        return jsonify({'error': 'Please provide a location.'}), 400
    
    # Construct a search query.
    query = f"advocates in {location}"
    if specialty:
        query = f"{specialty} {query}"

    # Use a mock function to simulate fetching data
    advocates = search_advocates_data(location, query)
    
    return jsonify({'advocates': advocates})

if __name__ == '__main__':
    # Initialize RAG chain on startup if a Chroma DB exists
    initialize_rag_chain()
    # Run the Flask app
    app.run(debug=True)