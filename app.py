from flask import Flask, request, render_template, send_file, url_for
import os
import cv2
from werkzeug.utils import secure_filename
from FULLSYSTEM import classifyDisease, lesionDetection
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from datetime import datetime
from FULLSYSTEM import *
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'
PDF_FOLDER = 'static/pdfs/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['PDF_FOLDER'] = PDF_FOLDER

# Create folders if not exist
for folder in [UPLOAD_FOLDER, RESULT_FOLDER, PDF_FOLDER]:
    os.makedirs(folder, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Enhanced grade mappings with detailed descriptions
GRADE_INFO = {
    0: (
        "No Diabetic Retinopathy", 
        "No signs of diabetic retinopathy detected. Continue regular annual eye examinations.",
        "DR is not present at this stage. The retina appears healthy with no visible abnormalities."
    ),
    1: (
        "Mild Non-Proliferative Diabetic Retinopathy (NPDR)", 
        "Early stage detected. Monitor blood sugar levels closely and follow up with your ophthalmologist in 9-12 months.",
        "Characterized by the presence of at least one microaneurysm (small blood vessel bulges) in the retina. These tiny red dots are the earliest clinically visible sign of diabetic retinopathy."
    ),
    2: (
        "Moderate Non-Proliferative Diabetic Retinopathy", 
        "Progressive stage detected. Schedule a comprehensive eye examination within 6 months and maintain strict glycemic control.",
        "More microaneurysms are present along with other signs such as dot and blot hemorrhages, hard exudates (lipid deposits), and cotton wool spots (nerve fiber layer infarcts)."
    ),
    3: (
        "Severe Non-Proliferative Diabetic Retinopathy", 
        "Advanced stage detected. Immediate consultation with an ophthalmologist is necessary. Risk of progression to PDR is significant.",
        "Characterized by the 4-2-1 rule: extensive intraretinal hemorrhages in 4 quadrants, venous beading in 2+ quadrants, or intraretinal microvascular abnormalities (IRMA) in 1+ quadrant. The blood vessels are severely damaged and blocked."
    ),
    4: (
        "Proliferative Diabetic Retinopathy (PDR)", 
        "Critical stage detected. Urgent specialist treatment required! High risk of vision loss without immediate intervention.",
        "New abnormal blood vessels grow on the retina or optic disc (neovascularization) due to severe ischemia. These vessels are fragile and can leak blood, causing vitreous hemorrhage, and may lead to retinal detachment and blindness if untreated."
    )
}

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            # Predict class
            grade = classifyDisease(clsmodel, save_path)

            # Predict segmentation (returns cv2 image)
            segimage = lesionDetection(save_path, segmodel)

            # Save segmented image
            seg_filename = os.path.splitext(filename)[0] + '_segmented.png'
            seg_save_path = os.path.join(app.config['RESULT_FOLDER'], seg_filename)
            cv2.imwrite(seg_save_path, segimage)

            # Get grade name, advice, and description
            grade_name, advice, description = GRADE_INFO.get(grade, ("Unknown", "Consult doctor.", "Unable to determine severity."))

            # Generate PDF report
            pdf_filename = os.path.splitext(filename)[0] + '_report.pdf'
            pdf_path = os.path.join(app.config['PDF_FOLDER'], pdf_filename)
            
            # Pass original and segmented image paths to PDF generator
            original_image_path = save_path
            segmented_image_path = seg_save_path
            
            generate_pdf_report(pdf_path, filename, grade, grade_name, advice, description, original_image_path, segmented_image_path)

            # Pass everything to template
            return render_template('result.html',
                                   original_image=url_for('static', filename='uploads/' + filename),
                                   segmented_image=url_for('static', filename='results/' + seg_filename),
                                   grade=grade,
                                   grade_name=grade_name,
                                   advice=advice,
                                   description=description,
                                   pdf_url=url_for('download_file', filename='pdfs/' + pdf_filename))
    return render_template('index.html')

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_file(os.path.join('static', filename), as_attachment=True)

def generate_pdf_report(filepath, image_name, grade, grade_name, advice, description, original_image_path, segmented_image_path):
    doc = SimpleDocTemplate(filepath, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Create custom styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=20,
        alignment=1,  # Center alignment
        spaceAfter=20
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.darkblue,
        spaceAfter=10
    )
    
    normal_text = ParagraphStyle(
        'BodyText',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=8
    )
    
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Italic'],
        fontSize=9,
        textColor=colors.darkgrey,
        alignment=1  # Center alignment
    )
    
    # Add title
    title = Paragraph("<b>DIABETIC RETINOPATHY ASSESSMENT REPORT</b>", title_style)
    story.append(title)
    
    # Add current date
    current_date = datetime.now().strftime("%B %d, %Y")
    date_text = Paragraph(f"<b>Report Date:</b> {current_date}", normal_text)
    story.append(date_text)
    story.append(Spacer(1, 0.2*inch))
    
    # Patient information section
    story.append(Paragraph("<b>PATIENT INFORMATION</b>", section_heading))
    story.append(Paragraph(f"<b>Image Filename:</b> {image_name}", normal_text))
    story.append(Paragraph("<b>Patient ID:</b> For clinical use, please enter patient information", normal_text))
    story.append(Spacer(1, 0.2*inch))
    
    # Results section
    story.append(Paragraph("<b>ANALYSIS RESULTS</b>", section_heading))
    
    # Create severity indicator
    severity_table_data = [[f"Grade {grade}: {grade_name}"]]
    severity_table = Table(severity_table_data, colWidths=[6*inch])
    
    # Color code based on severity
    if grade == 0:
        bg_color = colors.lightgreen
    elif grade == 1:
        bg_color = colors.lightblue
    elif grade == 2:
        bg_color = colors.yellow
    elif grade == 3:
        bg_color = colors.orange
    else:  # grade 4
        bg_color = colors.red
    
    severity_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg_color),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(severity_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Detailed description
    story.append(Paragraph("<b>Clinical Description:</b>", normal_text))
    story.append(Paragraph(description, normal_text))
    story.append(Spacer(1, 0.1*inch))
    
    # Medical advice
    story.append(Paragraph("<b>Medical Advice:</b>", normal_text))
    story.append(Paragraph(advice, normal_text))
    story.append(Spacer(1, 0.3*inch))
    
    # Add images
    story.append(Paragraph("<b>RETINAL IMAGES</b>", section_heading))
    
    # Create a table for side-by-side images
    image_width = 2.0*inch
    image_height = 2.0*inch
    
    # Resize images to fit properly
    try:
        orig_img = Image(original_image_path, width=image_width, height=image_height)
        seg_img = Image(segmented_image_path, width=image_width, height=image_height)
        
        image_table_data = [
            [Paragraph("<b>Original Image</b>", normal_text), Paragraph("<b>AI-Processed Image with Lesions</b>", normal_text)],
            [orig_img, seg_img]
        ]
        
        image_table = Table(image_table_data, colWidths=[3*inch, 3*inch])
        image_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(image_table)
    except Exception as e:
        story.append(Paragraph(f"Error loading images: {str(e)}", normal_text))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Information about diabetic retinopathy
    story.append(Paragraph("<b>ABOUT DIABETIC RETINOPATHY</b>", section_heading))
    
    dr_info = """
    <b>What is Diabetic Retinopathy?</b><br/>
    Diabetic retinopathy is a diabetes complication that affects the eyes. It's caused by damage to the blood vessels in the retina (the light-sensitive tissue at the back of the eye). Over time, too much sugar in the blood can lead to blockage of the tiny blood vessels that nourish the retina, cutting off its blood supply. As a result, the eye attempts to grow new blood vessels, but these don't develop properly.
    <br/><br/>
    <b>Stages of Diabetic Retinopathy:</b><br/>
    <b>1. No Diabetic Retinopathy (Grade 0):</b> No abnormalities detected.<br/>
    <b>2. Mild NPDR (Grade 1):</b> Microaneurysms (small swellings in blood vessels) appear.<br/>
    <b>3. Moderate NPDR (Grade 2):</b> Some blood vessels that nourish the retina become blocked.<br/>
    <b>4. Severe NPDR (Grade 3):</b> Many blood vessels are blocked, depriving several areas of the retina of blood supply.<br/>
    <b>5. Proliferative DR (Grade 4):</b> New, abnormal blood vessels grow (proliferate) in the retina, which can leak and cause severe vision problems.
    """
    story.append(Paragraph(dr_info, normal_text))
    story.append(Spacer(1, 0.2*inch))
    
    # Disclaimer
    story.append(Paragraph("<b>DISCLAIMER</b>", section_heading))
    disclaimer_text = """
    This report is generated by an automated Diabetic Retinopathy Detection System using artificial intelligence 
    algorithms to analyze retinal images. The results provided are meant to assist healthcare professionals 
    and should NOT be used as the sole basis for medical decisions. This report MUST be reviewed and verified by a 
    qualified ophthalmologist or healthcare professional before any treatment decisions are made. Early detection 
    and treatment of diabetic retinopathy can significantly reduce the risk of vision loss.
    """
    story.append(Paragraph(disclaimer_text, normal_text))
    
    # Footer
    story.append(Spacer(1, 0.5*inch))
    footer_text = "© Diabetic Retinopathy Detection System | Generated automatically | Page 1"
    story.append(Paragraph(footer_text, disclaimer_style))
    
    # Build the PDF
    doc.build(story)

if __name__ == "__main__":
    app.run(debug=True)