import streamlit as st
import fitz  # PyMuPDF pour lire les PDF
import google.generativeai as genai
import os
import random
from dotenv import load_dotenv
import tempfile
import pandas as pd
import base64
import io
import re
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from PIL import Image

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Configuration de l'API Gemini
def setup_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.sidebar.error("Clé API Gemini non trouvée dans le fichier .env")
        st.sidebar.info("Veuillez créer un fichier .env avec GEMINI_API_KEY=votre_clé_api")
        st.stop()
    
    # Configuration de l'API
    genai.configure(api_key=api_key)
    
    # Utiliser le modèle spécifié
    MODEL_NAME = "gemini-2.0-flash"
    
    try:
        # Création du modèle sans messages de débogage
        model = genai.GenerativeModel(MODEL_NAME)
        return model
    
    except Exception as e:
        st.sidebar.error(f"Erreur lors de la configuration du modèle Gemini: {str(e)}")
        st.stop()

def extract_text_from_pdf(file_path):
    """Extraire le texte d'un fichier PDF à partir d'un chemin."""
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Erreur lors de l'extraction du texte: {str(e)}")
        return ""

def generate_questions_with_gemini(model, text, num_questions=5, question_type="multiple_choice", difficulty_distribution=None):
    """Générer des questions en utilisant l'API Gemini avec une distribution de difficulté personnalisée."""
    
    # Limiter la taille du texte pour éviter de dépasser les limites de l'API
    max_tokens = 30000  # Ajustez selon les limitations de Gemini
    text = text[:max_tokens] if len(text) > max_tokens else text
    
    # Utiliser la distribution fournie par l'utilisateur
    if difficulty_distribution:
        easy_percent = difficulty_distribution['facile']
        medium_percent = difficulty_distribution['moyen']
        hard_percent = difficulty_distribution['difficile']
        distribution_text = f"{easy_percent}% faciles, {medium_percent}% moyennes et {hard_percent}% difficiles"
    else:
        # Distribution par défaut si aucune n'est fournie
        easy_percent = 40
        medium_percent = 40
        hard_percent = 20
        distribution_text = "40% faciles, 40% moyennes et 20% difficiles"
    
    # Calculer le nombre de questions par niveau de difficulté
    total_q = num_questions
    easy_q = round(total_q * easy_percent / 100)
    hard_q = round(total_q * hard_percent / 100)
    medium_q = total_q - easy_q - hard_q
    
    # Ajuster si nécessaire
    if medium_q < 0:
        medium_q = 0
        if easy_q > hard_q:
            easy_q = total_q - hard_q
        else:
            hard_q = total_q - easy_q
    
    # Instructions spécifiques pour le format selon le type de question
    format_instructions = ""
    if question_type == "multiple_choice":
        format_instructions = """
        Format pour les questions à choix multiples:
        
        1. Liste d'abord toutes les questions numérotées (1, 2, 3, etc.) avec un format clair et structuré
        
        2. Pour chaque question:
           - Écris clairement l'énoncé de la question
           - Présente les options de réponse sous forme de liste avec A), B), C), D) - une option par ligne
           - Laisse une ligne vide entre chaque question pour améliorer la lisibilité
        
        3. Dans une section "RÉPONSES" séparée à la fin, indique la réponse correcte pour chaque question:
           - Format: "Question 1: A", "Question 2: C", etc.
        """
    elif question_type == "vrai_faux":
        format_instructions = """
        Format pour les questions vrai/faux:
        
        1. Liste d'abord toutes les affirmations numérotées (1, 2, 3, etc.)
        2. Dans une section "RÉPONSES" séparée à la fin, indique si chaque affirmation est vraie ou fausse:
           - Format: "Affirmation 1: Vrai", "Affirmation 2: Faux", etc.
        """
    else:  # ouvert
        format_instructions = """
        Format pour les questions ouvertes:
        
        1. Liste d'abord toutes les questions numérotées (1, 2, 3, etc.)
        2. Dans une section "RÉPONSES SUGGÉRÉES" à la fin, propose des réponses possibles pour chaque question
        """
    
    # Prompt modifié pour utiliser la distribution de difficulté sélectionnée par l'utilisateur
    prompt = f"""
    Analyse le texte suivant et génère {num_questions} questions de type {question_type} avec une distribution de difficulté spécifique.
    
    Distribution de difficulté:
    - {easy_q} questions faciles (questions directes sur des informations explicites du texte)
    - {medium_q} questions moyennes (questions nécessitant une compréhension plus approfondie du texte)
    - {hard_q} questions difficiles (questions nécessitant une analyse, synthèse ou inférence à partir du texte)
    
    EXIGENCE IMPORTANTE: Les questions doivent être présentées dans un ordre TOTALEMENT ALÉATOIRE. 
    Les niveaux de difficulté doivent être complètement mélangés, sans aucun regroupement ou motif prévisible.
    
    Pour chaque question:
    - Assure-toi qu'elle soit basée sur le contenu du texte fourni
    - Évite les questions ambiguës ou trop générales
    - Ne précise pas la difficulté de la question dans l'énoncé
    - Assure-toi que la sélection des difficultés (facile, moyenne, difficile) apparaît dans un ordre véritablement aléatoire
    
    {format_instructions}
    
    TEXTE:
    {text}
    
    QUESTIONNAIRE ({question_type}):
    """
    
    try:
        # Configuration de la génération
        generation_config = {
            "temperature": 0.9,  # Température plus élevée pour plus d'aléatoire
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
        
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        return response.text, distribution_text
    except Exception as e:
        error_msg = f"Erreur lors de la génération des questions: {str(e)}"
        st.error(error_msg)
        return error_msg, None

def modify_questionnaire(model, current_questions, instructions):
    """Modifier le questionnaire selon les instructions de l'utilisateur."""
    prompt = f"""
    Voici un questionnaire existant:
    
    {current_questions}
    
    Instructions pour modifier ce questionnaire:
    {instructions}
    
    Veuillez fournir le questionnaire modifié en suivant ces instructions.
    Conservez la même structure avec les questions d'abord et les réponses à la fin.
    Ne mentionnez pas les instructions ou les changements demandés dans le résultat final.
    Ne pas inclure de commentaires ou d'explications sur les modifications dans le document final.
    Fournissez uniquement le questionnaire modifié.
    """
    
    try:
        # Configuration de la génération
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        return response.text
    except Exception as e:
        error_msg = f"Erreur lors de la modification du questionnaire: {str(e)}"
        st.error(error_msg)
        return error_msg

def create_pdf_bytes(questions_text, title="Questionnaire généré", logo_path=None):
    """
    Crée un PDF à partir du texte des questions en utilisant ReportLab qui supporte l'UTF-8.
    Inclut un logo d'entreprise si fourni.
    """
    try:
        # Utiliser ReportLab qui est plus robuste pour l'encodage
        from io import BytesIO
        
        # Créer un buffer pour le PDF
        buffer = BytesIO()
        
        # Configuration du document
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Créer un style personnalisé pour le titre
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            alignment=TA_CENTER,
            spaceAfter=12
        )
        
        # Styles pour différentes parties du questionnaire
        normal_style = styles['Normal']
        heading_style = styles['Heading2']
        option_style = ParagraphStyle(
            'Option',
            parent=styles['Normal'],
            leftIndent=20
        )
        
        # Préparer le contenu
        content = []
        
        # Ajouter le logo si disponible
        if logo_path and os.path.exists(logo_path):
            # Ajouter le logo en haut du document
            logo = RLImage(logo_path, width=100, height=50)  # Ajustez les dimensions selon votre logo
            content.append(logo)
            content.append(Spacer(1, 12))
        
        # Ajouter le titre
        content.append(Paragraph(title, title_style))
        content.append(Spacer(1, 12))
        
        # Traiter les lignes du questionnaire
        lines = questions_text.split("\n")
        in_question = False
        in_options = False
        
        for line in lines:
            line = line.strip()
            
            if not line:
                content.append(Spacer(1, 6))
                continue
                
            if "RÉPONSES" in line or "RÉPONSES SUGGÉRÉES" in line:
                # Section de réponses
                content.append(Spacer(1, 12))
                content.append(Paragraph(f'<b>{line}</b>', heading_style))
                content.append(Spacer(1, 6))
                in_question = False
                in_options = False
            elif line.strip() and line[0].isdigit() and any(c in line[1:3] for c in [".", ")", ":"]):
                # Numéro de question
                content.append(Paragraph(f'<b>{line}</b>', normal_style))
                in_question = True
                in_options = False
            elif line.strip() and line.startswith(("A)", "B)", "C)", "D)")):
                # Options de réponse
                content.append(Paragraph(line, option_style))
                in_options = True
                in_question = False
            elif line.strip().startswith(("Question", "Affirmation")) and ":" in line:
                # Ligne de réponse
                content.append(Paragraph(f'<b>{line}</b>', normal_style))
                in_question = False
                in_options = False
            else:
                # Texte normal
                if in_question or in_options:
                    content.append(Paragraph(line, option_style if in_options else normal_style))
                else:
                    content.append(Paragraph(line, normal_style))
        
        # Générer le PDF
        doc.build(content)
        
        # Récupérer les données du buffer
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes
    
    except Exception as e:
        st.error(f"Erreur lors de la création du PDF: {str(e)}")
        # Solution de repli - texte brut
        return questions_text.encode('utf-8')

def convert_to_excel(questions_text, question_type):
    """Convertir le questionnaire en format Excel."""
    # Initialiser des listes pour stocker les questions et réponses
    questions = []
    answers = []
    
    # Diviser le texte en sections (questions et réponses)
    sections = questions_text.split("RÉPONSES")
    if len(sections) < 2:
        sections = questions_text.split("RÉPONSES SUGGÉRÉES")
    
    questions_section = sections[0]
    answers_section = "RÉPONSES" + sections[1] if len(sections) > 1 else ""
    
    # Traiter selon le type de question
    if question_type == "multiple_choice":
        # Extraire les questions et options
        question_blocks = []
        current_block = []
        
        for line in questions_section.split('\n'):
            line = line.strip()
            # CORRECTION: Utiliser une expression régulière pour détecter les numéros de question
            # Ceci fonctionnera pour n'importe quel nombre de chiffres (1, 2, 10, 100, etc.)
            if line and re.match(r'^\d+[.)]', line):
                if current_block:
                    question_blocks.append("\n".join(current_block))
                current_block = [line]
            elif line:
                current_block.append(line)
        
        if current_block:
            question_blocks.append("\n".join(current_block))
        
        # Extraire questions et options
        for block in question_blocks:
            if not block.strip():
                continue
            
            lines = block.split("\n")
            question_text = lines[0]
            options = [line for line in lines[1:] if line.strip().startswith(("A)", "B)", "C)", "D)"))]
            
            questions.append({
                "Question": question_text,
                "Option A": next((opt.replace("A)", "").strip() for opt in options if opt.startswith("A)")), ""),
                "Option B": next((opt.replace("B)", "").strip() for opt in options if opt.startswith("B)")), ""),
                "Option C": next((opt.replace("C)", "").strip() for opt in options if opt.startswith("C)")), ""),
                "Option D": next((opt.replace("D)", "").strip() for opt in options if opt.startswith("D)")), "")
            })
        
        # Extraire les réponses
        # CORRECTION: Utiliser une expression régulière pour extraire les réponses
        answer_lines = [line.strip() for line in answers_section.split('\n') 
                       if re.search(r'Question\s+\d+\s*:', line)]
        
        for line in answer_lines:
            parts = line.split(":")
            if len(parts) >= 2:
                answers.append(parts[1].strip())
    
    elif question_type == "vrai_faux":
        # Extraire les affirmations avec regex
        question_lines = [line.strip() for line in questions_section.split('\n') 
                         if line.strip() and re.match(r'^\d+[.)]', line.strip())]
        
        for line in question_lines:
            questions.append({"Affirmation": line})
        
        # Extraire les réponses avec regex
        answer_lines = [line.strip() for line in answers_section.split('\n') 
                       if re.search(r'Affirmation\s+\d+\s*:', line)]
        
        for line in answer_lines:
            parts = line.split(":")
            if len(parts) >= 2:
                answers.append(parts[1].strip())
    
    else:  # questions ouvertes
        # Extraire les questions avec regex
        question_lines = [line.strip() for line in questions_section.split('\n') 
                         if line.strip() and re.match(r'^\d+[.)]', line.strip())]
        
        for line in question_lines:
            questions.append({"Question": line})
        
        # Extraire les réponses suggérées
        answers_section_lines = answers_section.split('\n')
        current_answer = ""
        current_q_num = 0
        
        # Améliorer la détection des numéros de question dans les réponses
        for line in answers_section_lines:
            match = re.search(r'^\d+[.)]', line.strip())
            if line.strip() and match:
                if current_answer and current_q_num > 0:
                    answers.append(current_answer)
                current_answer = line.split(":", 1)[1].strip() if ":" in line else ""
                current_q_num += 1
            elif line.strip() and current_q_num > 0:
                current_answer += " " + line.strip()
        
        if current_answer:
            answers.append(current_answer)
    
    # Créer le DataFrame pour l'Excel
    if question_type == "multiple_choice":
        df_questions = pd.DataFrame(questions)
        if len(answers) == len(questions):
            df_questions["Réponse correcte"] = answers
    elif question_type == "vrai_faux":
        df_questions = pd.DataFrame(questions)
        if len(answers) == len(questions):
            df_questions["Réponse"] = answers
    else:  # questions ouvertes
        df_questions = pd.DataFrame(questions)
        if len(answers) == len(questions):
            df_questions["Réponse suggérée"] = answers
    
    # Créer le fichier Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Ajouter les questions
        df_questions.to_excel(writer, index=False, sheet_name='Questionnaire')
        
        # Formatage
        workbook = writer.book
        worksheet = writer.sheets['Questionnaire']
        
        # Ajuster la largeur des colonnes
        for i, col in enumerate(df_questions.columns):
            column_width = max(df_questions[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(i, i, column_width)
    
    output.seek(0)
    return output.getvalue()

st.set_page_config(
        page_title="Générateur de Questionnaires",
        layout="wide"
    )
    
def main():
    # Configuration de la page avec une mise en page large
 
    # Initialiser les variables de session si elles n'existent pas
    if 'questions' not in st.session_state:
        st.session_state.questions = ""
    if 'distribution_text' not in st.session_state:
        st.session_state.distribution_text = ""
    if 'question_type' not in st.session_state:
        st.session_state.question_type = "multiple_choice"
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'logo_path' not in st.session_state:
        st.session_state.logo_path = None
    
    # Mise en page du haut avec le logo
    header_col1, header_col2 = st.columns([1, 5])
    
    with header_col1:
    
        
        # Option 1: Saisie directe du chemin du logo
        logo_path_input = r"download.png"
        
        # Option 2: Téléchargement du logo
    
        # Traitement du logo selon l'option choisie
        if logo_path_input and os.path.exists(logo_path_input):
            # Utiliser le chemin fourni
            st.session_state.logo_path = logo_path_input
            try:
                logo_image = Image.open(logo_path_input)
                st.image(logo_image, width=150)
            except Exception as e:
                st.error(f"Impossible de charger l'image: {str(e)}")
    with header_col2:
        st.title("Générateur de Questionnaires")
    
    # Barre latérale pour les paramètres
    with st.sidebar:
        # Afficher le logo dans la barre latérale s'il est disponible
        if st.session_state.logo_path and os.path.exists(st.session_state.logo_path):
            try:
                sidebar_logo = Image.open(st.session_state.logo_path)
                st.image(sidebar_logo, width=150)
            except Exception as e:
                st.error(f"Erreur d'affichage du logo dans la barre latérale: {str(e)}")
        
        st.title("Paramètres")
        
        # Upload de fichier
        st.header("Étape 1: Téléchargez votre fichier")
        uploaded_file = st.file_uploader("Choisissez un fichier PDF", key="pdf_uploader")
        
        if uploaded_file is not None:
            # Vérifier manuellement l'extension du fichier
            file_name = uploaded_file.name
            file_extension = os.path.splitext(file_name)[1].lower()
            
            if file_extension != '.pdf':
                st.error(f"Le fichier doit être un PDF. Vous avez téléchargé un fichier avec l'extension {file_extension}")
            else:
                st.success(f"Fichier téléchargé: {file_name}")
                
                # Options pour la génération de questions
                st.header("Étape 2: Paramètres du questionnaire")
                
                # Champ numérique pour le nombre de questions
                num_questions = st.number_input(
                    "Nombre de questions à générer", 
                    min_value=1, 
                    max_value=50, 
                    value=5,
                    step=1
                )
                
                # Type de questions
                question_type = st.selectbox(
                    "Type de questions",
                    options=["multiple_choice", "vrai_faux", "ouvert"],
                    format_func=lambda x: {
                        "multiple_choice": "Choix multiples", 
                        "vrai_faux": "Vrai ou Faux", 
                        "ouvert": "Questions ouvertes"
                    }[x]
                )
                
                # Nouvelle section pour la distribution de difficulté
                st.header("Étape 3: Distribution de difficulté")
                st.write("Choisissez le niveau de difficulté du questionnaire :")
                
                # Options prédéfinies de distribution de difficulté (seulement 3)
                difficulty_options = {
                    "Facile": {"facile": 60, "moyen": 30, "difficile": 10},
                    "Moyen": {"facile": 40, "moyen": 40, "difficile": 20},
                    "Difficile": {"facile": 20, "moyen": 40, "difficile": 40}
                }
                
                # Sélecteur de distribution
                selected_difficulty = st.selectbox(
                    "Niveau de difficulté général",
                    options=list(difficulty_options.keys()),
                    index=1,  # Par défaut "Moyen"
                    help="Sélectionnez le niveau de difficulté global que vous souhaitez pour votre questionnaire"
                )
                
                # Afficher la distribution correspondante
                selected_distribution = difficulty_options[selected_difficulty]
                facile_percent = selected_distribution["facile"]
                moyen_percent = selected_distribution["moyen"]
                difficile_percent = selected_distribution["difficile"]
                
                # Affichage informatif de la distribution sélectionnée
                st.info(f"📊 **Distribution sélectionnée** : {facile_percent}% facile, {moyen_percent}% moyen, {difficile_percent}% difficile")
                
                # Explication des niveaux
                with st.expander("ℹ️ Explication des niveaux de difficulté"):
                    st.write("""
                    **Questions faciles** : Questions directes basées sur des informations explicites du texte
                    
                    **Questions moyennes** : Questions nécessitant une compréhension plus approfondie du contenu
                    
                    **Questions difficiles** : Questions nécessitant une analyse, synthèse ou inférence à partir du texte
                    """)
                
                # Le bouton n'est plus désactivé car toutes les distributions sont valides
                generate_button_disabled = False
                
                if st.button("Générer les questions", disabled=generate_button_disabled):
                    # Initialiser le modèle Gemini
                    model = setup_gemini()
                    
                    # Préparer la distribution de difficulté
                    difficulty_distribution = {
                        'facile': facile_percent,
                        'moyen': moyen_percent,
                        'difficile': difficile_percent
                    }
                    
                    # Sauvegarde temporaire du fichier
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Extraction du texte
                    with st.spinner("Extraction du texte du PDF..."):
                        pdf_text = extract_text_from_pdf(tmp_path)
                        
                        # Supprimer le fichier temporaire
                        os.unlink(tmp_path)
                        
                        if not pdf_text:
                            st.error("Impossible d'extraire le texte du PDF. Veuillez vérifier le format du fichier.")
                            st.stop()
                    
                    # Génération des questions avec Gemini
                    with st.spinner(f"Génération des questions..."):
                        questions, distribution_text = generate_questions_with_gemini(
                            model, 
                            pdf_text, 
                            num_questions, 
                            question_type,
                            difficulty_distribution
                        )
                        
                        # Sauvegarder dans la session
                        st.session_state.questions = questions
                        st.session_state.distribution_text = distribution_text
                        st.session_state.question_type = question_type
    
    # Contenu principal - tout en une seule page
    if st.session_state.questions:
        st.header("Questionnaire généré")
        
        # Afficher la distribution de difficulté sélectionnée par l'utilisateur
        st.info(f"Distribution de difficulté sélectionnée: {st.session_state.distribution_text}")
        
        # Afficher les questions
        st.markdown(st.session_state.questions)
        
        # Options de téléchargement
        st.subheader("Télécharger le questionnaire")
        
        col1, col2 = st.columns(2)
        
        # Téléchargement PDF
        with col1:
            try:
                pdf_bytes = create_pdf_bytes(
                    st.session_state.questions, 
                    "Questionnaire généré", 
                    st.session_state.logo_path
                )
                st.download_button(
                    label="Télécharger en PDF",
                    data=pdf_bytes,
                    file_name="questionnaire.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"Impossible de créer le PDF: {str(e)}")
                st.info("Essayez de télécharger au format Excel à la place.")
        
        # Téléchargement Excel
        with col2:
            try:
                excel_bytes = convert_to_excel(
                    st.session_state.questions, 
                    st.session_state.question_type
                )
                st.download_button(
                    label="Télécharger en Excel",
                    data=excel_bytes,
                    file_name="questionnaire.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"Impossible de créer le fichier Excel: {str(e)}")
                st.info("Essayez de télécharger au format texte à la place.")
                st.download_button(
                    label="Télécharger en texte",
                    data=st.session_state.questions,
                    file_name="questionnaire.txt",
                    mime="text/plain"
                )
        
        # Séparateur avant le chatbot
        st.markdown("---")
        
        # Section du chatbot pour modifier le questionnaire
        st.header("Modifier le questionnaire")
        
        # Afficher l'historique des conversations
        chat_container = st.container()
        with chat_container:
            for i, (role, content) in enumerate(st.session_state.chat_history):
                if role == "user":
                    st.markdown(f"**Vous:** {content}")
                else:
                    st.markdown(f"**Assistant:** {content}")
        
        # Zone de texte pour saisir des modifications
        user_input = st.text_area("Entrez vos instructions pour modifier le questionnaire:", 
                                  placeholder="Par exemple: 'Rendre la question 3 plus difficile' ou 'Ajouter une nouvelle question sur...'")
        
        if st.button("Envoyer"):
            if user_input:
                # Ajouter l'entrée utilisateur à l'historique
                st.session_state.chat_history.append(("user", user_input))
                
                # Initialiser le modèle Gemini si nécessaire
                model = setup_gemini()
                
                # Appliquer les modifications
                with st.spinner("Modification du questionnaire en cours..."):
                    modified_questions = modify_questionnaire(
                        model,
                        st.session_state.questions,
                        user_input
                    )
                    
                    # Mettre à jour le questionnaire
                    st.session_state.questions = modified_questions
                    
                    # Ajouter la réponse à l'historique
                    st.session_state.chat_history.append(("assistant", "J'ai modifié le questionnaire selon vos instructions."))
                
                # Utiliser st.rerun au lieu de st.experimental_rerun
                st.rerun()
    else:
        st.info("Utilisez la barre latérale pour télécharger un fichier PDF et générer un questionnaire.")
    
    # Nettoyer le fichier logo temporaire quand l'application se ferme
    def cleanup():
        # Ne pas supprimer le logo si c'est un chemin d'accès direct
        if (st.session_state.logo_path and os.path.exists(st.session_state.logo_path) and 
            (logo_path_input is None or st.session_state.logo_path != logo_path_input)):
            try:
                os.unlink(st.session_state.logo_path)
            except:
                pass
            
    # Enregistrer la fonction de nettoyage pour qu'elle soit exécutée à la fermeture
    import atexit
    atexit.register(cleanup)
    
if __name__ == "__main__":
    main()
