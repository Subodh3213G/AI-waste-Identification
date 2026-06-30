# Internship Mentorship Handbook: EcoSort AI - Waste Classifier & Sorting Assistant

**Project Name:** AI-Powered Waste Identification  
**Assigned Student:** Subodh Yadav  
**Mentorship Focus:** Learn → Build → Integrate → Test → Document → Present  

---

## 1. Project Overview

### Problem Statement
Inadequate waste segregation at source is the leading cause of land pollution, ocean garbage accumulation, and recycling failures. Municipalities and recycling facilities spend immense resources manually sorting garbage. Furthermore, everyday citizens struggle to identify which products are recyclable, compostable, or hazardous. An automated image-based sorting assistant can guide individuals on how to dispose of items correctly.

### Project Goal
The goal of this project is to build **EcoSort AI**, a web-based image recognition application using Python, Streamlit, TensorFlow/Keras, and the Google Gemini API. The application will allow users to upload an image of a waste item, classify it into a major waste category (Organic, Recyclable, Hazardous, E-waste) using a lightweight pre-trained Deep Learning model (MobileNetV2), and query the Gemini API to generate tailored eco-disposal instructions, environmental impact metrics, and carbon offset suggestions.

### Real-world Applications
* **Smart Disposal Kiosks:** Integrating camera systems onto public trash bins to automatically open correct segregation lids.
* **Smart Home Assistant:** Providing household users with an app to scan packaging barcodes or labels before recycling.

### Why This Project Matters
This project introduces interns to **Deep Learning Image Classification (Computer Vision)** and cloud integrations. Interns will master image preprocessing math (resizing, normalization, tensor operations), load and execute inference on pre-trained Convolutional Neural Networks (CNNs), and interface classified outputs with LLM advisory agents.

### Expected Final MVP
A Streamlit web application that lets users:
1. Upload an image of an object (JPG/PNG).
2. View the image on screen.
3. Check the classification model's prediction along with a confidence bar chart.
4. Review an **Eco-Disposal Guide Panel** detailing:
   * Correct Trash Bin color.
   * Recycling guidelines (e.g., wash before throwing).
   * Estimated decomposition time.
   * Carbon footprint reduction tips.

### Future Enhancements
* Object detection (YOLOv8) to locate and classify multiple waste objects in a single scene.
* Gamified green points tracker logging user recycling streaks in SQLite.

---

## 2. Difficulty Estimation

* **Level:** Intermediate  
* **Why:** The project is classified as *Intermediate* because:
  1. **Deep Learning Inference:** Loading and executing inference using TensorFlow/Keras tensors requires understanding dimensions (channel formatting, normalization steps).
  2. **Model Management:** Downloading weights files and caching model objects to prevent Streamlit reload lag requires proper backend structuring.

---

## 3. Skills Required

### Programming
* **Python:** Image stream operations, tensor calculations, dictionary configurations.
* **TensorFlow/Keras:** Loading model weights, compiling layers, and executing prediction arrays.
* **Streamlit:** File upload containers, image grids, and interactive markdown card layouts.
* **SQLite3:** Logging user classification histories and daily carbon offset scores.

### AI/ML
* **Deep Learning (CNNs):** Convolutional layers, pooling, softmax activation, MobileNetV2 architecture.
* **Image Preprocessing:** Tensor shaping, pixel normalizations, BGR/RGB channels adjustments.
* **Generative AI Advising:** Prompt engineering structures returning JSON-formatted recycling directions.

### Software Engineering
* **Git & GitHub:** Branch structures, commits logs, and repository setups.
* **Dependency Management:** Packaging large computer vision libraries (`tensorflow`, `opencv-python`, `pillow`).
* **Environment Configurations:** Protecting API credentials using environmental variables (`.env`).

---

## 4. Recommended Tech Stack

| Component | Technology | Why |
|---|---|---|
| **Frontend UI** | **Streamlit** | `st.file_uploader` and `st.image` widgets simplify building image-based web apps. |
| **Image Loader** | **Pillow (PIL) & NumPy** | Standard libraries to load image file streams and convert them into tensor matrices. |
| **Deep Learning** | **TensorFlow / Keras (MobileNetV2)** | MobileNetV2 is a lightweight CNN designed for mobile and embedded devices. It runs fast on standard CPUs, making it ideal for a 15-day student MVP. |
| **AI LLM API** | **Google Gemini Developer API (`gemini-2.5-flash`)** | Generous free tier, fast processing speed, and excellent at generating structured advisory reports. |
| **Visual Charts** | **Plotly Express** | Plots classification confidence probabilities on horizontal bar charts. |
| **Database Logs** | **SQLite3** | Registers user classification results and daily carbon savings logs. |

---

## 5. System Architecture

### Text-Based Architecture Flowchart

```text
                  +-----------------------------------+
                  |          User Uploads             |
                  |          Waste Item Image         |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  |        Streamlit Interface        |
                  |       (File Ingestion Widget)     |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  |        PIL / NumPy Loader         |
                  |    (Convert to RGB, Resize)       |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  |      MobileNetV2 CNN Model        |
                  |    (Predicts Waste Category)      |
                  +-----------------+-----------------+
                                    |
                 [Waste Class Name & Confidence Score]
                                    |
                                    v
                  +-----------------------------------+
                  |      Gemini AI Advisory API       |
                  |    (Generates Disposal Report)    |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  |       Streamlit Dashboard         |
                  |  (Renders Image + Disposal Guide) |
                  +-----------------------------------+
```

---

## 6. Development Modules

### Module 1: Image Uploader & Validator
* **Purpose:** Handle file uploads (PNG/JPG), validate image formats, and convert raw byte streams into PIL Image objects.
* **Inputs:** Uploaded image stream.
* **Outputs:** PIL Image object.
* **Dependencies:** `streamlit`, `pillow`

### Module 2: Image Tensor Preprocessor
* **Purpose:** Resize images for model consistency (224x224), expand dimensions to match batch size shapes, and normalize pixel values to [-1, 1] range.
* **Inputs:** PIL Image object.
* **Outputs:** Preprocessed NumPy tensor array.
* **Dependencies:** `tensorflow`, `numpy`

### Module 3: MobileNetV2 Classifier Client
* **Purpose:** Load the pre-trained MobileNetV2 model and run inference on the tensor image to return top predicted categories.
* **Inputs:** Preprocessed tensor array.
* **Outputs:** Class label string, confidence score list.
* **Dependencies:** `tensorflow`

### Module 4: Gemini Eco-Advisor Agent
* **Purpose:** Prompt Gemini to generate recycling directions and environmental impact details based on the predicted category.
* **Inputs:** Predicted class label.
* **Outputs:** Structured disposal instructions JSON.
* **Dependencies:** `google-generativeai`, `pydantic`

### Module 5: Database Logger
* **Purpose:** Log database entries tracking user classification results, timestamps, and carbon offset scores.
* **Inputs:** SQLite transaction commands.
* **Outputs:** Local database update logs.
* **Dependencies:** `sqlite3`

### Module 6: Dashboard Interface
* **Purpose:** Display original images, prediction probability charts, metrics panels, and expandable disposal recommendation cards.
* **Inputs:** Streamlit widget interactions.
* **Outputs:** Rendered dashboard pages.
* **Dependencies:** `streamlit`, `plotly`

---

## 7. What Should Be Built vs Reused

| Build Yourself | Reuse Existing |
|---|---|
| Image preprocessing functions | MobileNetV2 CNN deep learning layers |
| Fine-tuning classifiers (optional) | TensorFlow weights files and models loaders |
| Structured disposal recommendation prompts | Streamlit image viewer widgets |
| SQLite database schemas (logs in SQLite) | Plotly Express horizontal bar layouts |
| Dashboard UI states managers | Gemini API Python SDK client wrappers |

---

## 8. Pre-trained Models and APIs

* **MobileNetV2 (ImageNet Weights):** Pre-trained on 1,000 general categories (including plastic cups, bottles, boxes). Excellent for immediately classifying common household items without training from scratch.
* **Google Gemini API (`gemini-2.5-flash`):** Core LLM for generating custom recycling guides and environmental details.

---

## 9. Existing Open Source Projects to Study

* **TensorFlow Keras Image Classification Tutorials:** [Keras Docs](https://keras.io/api/applications/) - Best resource for understanding Keras application models.
* **Streamlit App Gallery (Computer Vision section):** Review dashboards displaying classification pipelines.
* **What NOT to copy:** Large industrial waste classification platforms with complex database integrations. Focus on building a lightweight, local Streamlit dashboard.

---

## 10. Data Sources

* **Kaggle Garbage Classification Dataset:**
  * Contains thousands of images classified into 6 categories (glass, paper, cardboard, plastic, metal, trash).
  * Search keywords: `garbage classification dataset kaggle`.
  * Cost: Free.
* **Target Job Outlines:** Mock categories listings stored in JSON files for immediate testing.

---

## 11. Research Papers for Reference

| Title | Year | Summary | Why Read It |
|---|---|---|---|
| *MobileNetV2: Inverted Residuals and Linear Bottlenecks* | 2018 | Outlines the architecture of MobileNetV2 and its efficiency on mobile devices. | Helps understand lightweight CNN pipelines. |
| *Deep Learning-based Waste Management Systems* | 2021 | Evaluates CNNs and object detectors for waste sorting automation. | Informs classification categories design. |

---

## 12. Self-Learning Resources

### Official Documentation
* **Keras Applications Guide:** [Keras API](https://keras.io/api/applications/mobilenet/) - Essential for loading MobileNetV2.
* **Streamlit Widgets Reference:** [Streamlit API](https://docs.streamlit.io/develop/api-reference) - Focus on uploader and layouts.

### YouTube Crash Courses
* **TensorFlow Image Classification in 1 Hour:** *Deep Learning Classification Crash Course* (by freeCodeCamp).
* **OpenCV Basics Crash Course:** search: "Python OpenCV crash course".

---

## 13. Industry Standard Folder Structure

```text
ecosort_classifier/
│
├── config/
│   └── .env.example        # Template for GEMINI_API_KEY
│
├── database/
│   ├── __init__.py
│   ├── db_manager.py       # SQLite connection logging classifications history
│   └── classifications.db  # Database file (ignored in .gitignore)
│
├── app/
│   ├── __init__.py
│   ├── main.py             # Streamlit application main runner
│   ├── preprocessor.py     # Image resizing and normalization scripts
│   ├── classifier.py       # TensorFlow CNN inference client
│   └── advisor.py          # Gemini API recycling advisor
│
├── tests/
│   └── test_image.py       # Test image preprocessors and tensor shapes
│
├── requirements.txt
├── .gitignore
├── README.md
└── run.py                  # CLI script launching streamlit (`streamlit run app/main.py`)
```

---

## 14. GitHub Milestones

1. **Milestone 1: Repository Setup & TensorFlow Integration**
   * Structure project directories, configure virtual environment, install TensorFlow and verify installation.
2. **Milestone 2: Keras Model Loading**
   * Write scripts to load MobileNetV2 and test inference on standard household items (e.g. plastic bottles).
3. **Milestone 3: Image Preprocessing Pipeline**
   * Implement functions to resize, normalize, and shape raw image streams into tensors.
4. **Milestone 4: Gemini API Integration**
   * Connect Gemini API. Write prompts that generate recycling directions and environmental impact details based on the predicted category.
5. **Milestone 5: SQLite Database Logger**
   * Set up SQLite databases registering user classification history and daily carbon offset logs.
6. **Milestone 6: Streamlit UI Implementation**
   * Design the dashboard layout, metrics cards, confidence bar charts, and expandable advisory report panels.
7. **Milestone 7: Unit Testing & Deployment**
   * Write tests for image handling, document the project, and record the demo video.

---

## 15. MVP Planning

### Must Have
* Image uploader (JPG/PNG).
* TensorFlow MobileNetV2 model classification inference.
* Confidence bar chart displaying prediction probabilities.
* Eco-Disposal Guide Panel displaying bin colors and recycling rules.
* Carbon footprint reduction tip cards.

### Nice to Have
* SQLite dashboard panel showing historical classifications history.
* Support for webcam snapshot captures directly from the web interface.

### Future Scope
* Object detection (YOLOv8) to locate and classify multiple waste objects in a single scene.

---

## 16. 15-Day Curriculum

| Day | Phase | Learning Goals | Topics to Study | Resources | What to Build | Expected Deliverable | Est. Time |
|---|---|---|---|---|---|---|---|
| **Day 1** | Phase 1 | Setup & Git | Project directory structures, virtual envs, Git branching. | Python Venv Guide. | Set up folders, configure Git, install dependencies. | Project directory with activated virtual environment. | 4 hrs |
| **Day 2** | Phase 1 | TensorFlow Intro | TensorFlow installations, checking GPU/CPU configurations. | TensorFlow installation guide. | Verify TensorFlow installation in your workspace environment. | Console log showing active TensorFlow version. | 5 hrs |
| **Day 3** | Phase 1 | Keras Model Loading | Keras application modules, model layers, loading weights. | Keras Applications docs. | Write Keras model loaders in `app/classifier.py`. | Script loading MobileNetV2 model details. | 5 hrs |
| **Day 4** | Phase 2 | Image Preprocessing | PIL Image loaders, resizing coordinates, aspect ratios. | PIL documentation. | Write image loaders in `app/preprocessor.py`. | Script reading and printing image shape details. | 6 hrs |
| **Day 5** | Phase 2 | Tensor Transformations | Dimensions expansion, normalization scales, channel layouts. | TensorFlow preprocessors. | Write preprocessing functions in `app/preprocessor.py`. | Script resizing and normalising images to 224x224 tensors. | 6 hrs |
| **Day 6** | Phase 2 | CNN Model Inference | Predict API methods, class mappings, probability lists. | Keras predictions guides. | Run model inference on standard waste objects. | Script outputting raw predictions and class mappings. | 6 hrs |
| **Day 7** | Phase 2 | Gemini API Connection | Prompts creation, system constraints, text evaluation. | Gemini Prompt guides. | Build recycling advisers in `app/advisor.py` using Gemini API. | Script outputting qualitative recycling guides. | 6 hrs |
| **Day 8** | Phase 2 | SQLite Database | SQL schemas, table creation, database query wrappers. | SQLite tutorial. | Build database schemas in `database/db_manager.py`. | Local SQLite DB saving classification histories. | 5 hrs |
| **Day 9** | Phase 2 | Streamlit Dashboard | Uploader widgets, image containers, metrics panels. | Streamlit API references. | Code main dashboard pages in `app/main.py` layouts. | Streamlit app rendering static page layouts. | 6 hrs |
| **Day 10**| Phase 3 | App Integration | Connecting image uploaders, CNN models, and Gemini API. | Python integration guides. | Link uploaders, CNN models, Gemini APIs, and gauges. | Dashboard updating classification outputs upon image upload. | 7 hrs |
| **Day 11**| Phase 3 | UI Visual Polish | Tabbed dashboard, indicator gauges. | Plotly Streamlit docs. | Add Plotly charts and sidebar parameter controls. | Dashboard displaying interactive charts and metrics. | 6 hrs |
| **Day 12**| Phase 3 | Performance Tuning | Caching loaders, optimization, managing memory leaks. | Streamlit performance. | Caching model loads to prevent reloading on every run. | High-performance dashboard app. | 6 hrs |
| **Day 13**| Phase 3 | System Testing | Unit testing structures, pytest asserts. | Pytest docs. | Write unit tests in `tests/test_image.py`. | Test runners passing all test cases. | 6 hrs |
| **Day 14**| Phase 4 | Project Documentation | Documentation layouts, writing README instructions. | README templates. | Write `README.md` and complete inline code comments. | Finished README explaining installation and design features. | 4 hrs |
| **Day 15**| Phase 4 | Presentation Day | Presentation design, delivery. | Presentation guides. | Design presentation slides detailing Architecture, SQL models, and CNN layouts. | Final slide deck. | 5 hrs |

---

## 17. Daily Output Expectations & Developer Code Guides

To ensure success, the intern must refer to the following code guide to run Keras CNN classification.

### Critical Developer Guide: Waste Ingestion and TensorFlow Classification
The following code snippet demonstrates how to load the MobileNetV2 model, preprocess an uploaded image, run classification, and query Gemini API to generate disposal instructions:

```python
# app/classifier.py
import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import google.generativeai as genai
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Step 1: Load pre-trained MobileNetV2 model (Cached to prevent Streamlit lag)
@tf.function
def get_mobilenet_model():
    # Load model pre-trained on ImageNet datasets
    return MobileNetV2(weights='imagenet')

# Step 2: Preprocess Image stream into standard CNN input tensor
def preprocess_image_tensor(pil_image: Image.Image) -> np.ndarray:
    # MobileNetV2 expects 224x224 RGB inputs
    img = pil_image.resize((224, 224))
    img_array = np.array(img)
    
    # Handle Grayscale inputs by repeating channels
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
        
    # Ensure correct channel dims (RGB)
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
        
    # Expand dimensions to Batch Shape: (1, 224, 224, 3)
    img_tensor = np.expand_dims(img_array, axis=0)
    
    # Scale pixels to MobileNetV2 expectations [-1, 1] range
    return preprocess_input(img_tensor.astype(np.float32))

# Step 3: Run Model Prediction
def classify_waste_item(pil_image: Image.Image):
    model = get_mobilenet_model()
    tensor = preprocess_image_tensor(pil_image)
    
    # Run prediction tensor
    preds = model.predict(tensor)
    
    # Decode top 3 predictions
    decoded = decode_predictions(preds, top=3)[0]
    
    # Extract highest confidence prediction
    top_class_id, top_label, top_conf = decoded[0]
    
    # Clean label underscores
    clean_label = top_label.replace("_", " ").capitalize()
    
    return clean_label, float(top_conf)

# Step 4: Query Gemini API for Disposal Details
class DisposalGuide(BaseModel):
    waste_category: str = Field(description="One of: Organic, Recyclable, Hazardous, E-waste")
    bin_color: str = Field(description="Trash bin color category")
    recycling_rules: list = Field(description="Top 3 recycling instructions")
    environmental_impact: str = Field(description="Environmental degradation impact statement")

def get_disposal_advisory(label: str) -> DisposalGuide:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""
    You are an expert environmental sorting specialist. The classified object is: {label}
    Evaluate how to dispose of this object. Output a structured JSON disposal guide.
    """
    
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=DisposalGuide
        )
    )
    
    data = json.loads(response.text)
    return DisposalGuide(**data)

# Example Mock Usage:
# pil_img = Image.open("bottle.jpg")
# label, conf = classify_waste_item(pil_img)
# guide = get_disposal_advisory(label)
# print(f"Classified: {label} (Conf: {conf})")
# print(f"Bin Color: {guide.bin_color}")
```

### Daily Expectations

#### Day 1
* **Learn:** Git workflows, PIP configurations, setup directories.
* **Build:** Setup local project directories and install dependencies.
* **Expected Output:** Local directory with virtual environment set up and active Git tracking.

#### Day 2
* **Learn:** TensorFlow Keras configurations.
* **Build:** Verify TensorFlow installation in your workspace environment.
* **Expected Output:** Console log showing active TensorFlow version.

#### Day 3
* **Learn:** Keras model loaders.
* **Build:** Write Keras model loaders in `app/classifier.py`.
* **Expected Output:** Script loading MobileNetV2 model details.

#### Day 4
* **Learn:** Image loading parameters.
* **Build:** Write image loaders in `app/preprocessor.py`.
* **Expected Output:** Script reading and printing image shape details.

#### Day 5
* **Learn:** Tensor resizing and normalizations.
* **Build:** Write preprocessing functions resizing image sizes.
* **Expected Output:** Script resizing and normalising images to 224x224 tensors.

#### Day 6
* **Learn:** CNN inference parameters.
* **Build:** Run model inference on standard waste objects.
* **Expected Output:** Script outputting raw predictions and class mappings.

#### Day 7
* **Learn:** Prompts optimization, Gemini API connections.
* **Build:** Build recycling advisers in `app/advisor.py`.
* **Expected Output:** Script outputting qualitative recycling guides.

#### Day 8
* **Learn:** SQLite write transactions.
* **Build:** Build database schemas in `database/db_manager.py`.
* **Expected Output:** Local SQLite DB saving classification histories.

#### Day 9
* **Learn:** Streamlit uploader layouts.
* **Build:** Code main dashboard pages in `app/main.py` layouts.
* **Expected Output:** Streamlit app rendering static page layouts.

#### Day 10
* **Learn:** Dashboard data connections.
* **Build:** Link uploaders, CNN models, Gemini APIs, and database logs.
* **Expected Output:** Dashboard updating classification outputs upon image upload.

#### Day 11
* **Learn:** Plotly chart widgets.
* **Build:** Add Plotly charts and sidebar parameter controls.
* **Expected Output:** Dashboard displaying interactive charts and metrics.

#### Day 12
* **Learn:** Caching deep learning models.
* **Build:** Caching model loads to prevent reloading on every run.
* **Expected Output:** High-performance dashboard app.

#### Day 13
* **Learn:** Pytest unit testing setups.
* **Build:** Write unit tests in `tests/test_image.py`.
* **Expected Output:** Test runners passing all test cases.

#### Day 14
* **Learn:** README documentation standards.
* **Build:** Write `README.md` and complete inline code comments.
* **Expected Output:** Finished README explaining installation and design features.

#### Day 15
* **Learn:** Presentation design, delivery.
* **Build:** Design presentation slides detailing architecture and milestones.
* **Expected Output:** Final presentation slide deck.

---

## 18. Final Deliverables

For evaluation, each intern must submit:
1. **Source Code:** Complete Python project directories including Keras wrappers.
2. **GitHub Repository:** Clean history of version control commits.
3. **README.md:** Explaining project context, setup instructions, and database details.
4. **Local Database:** SQLite DB file (`classifications.db`).
5. **Live Dashboard Link:** Deployed Streamlit Cloud URL (optional).
6. **Project Report:** Standard PDF detailing findings, model verification parameters (accuracy matrix), and carbon offset calculations.
7. **Demo Video:** A 3-minute video showing image uploading, classification charts, and dynamic disposal guides.
8. **Presentation Slides:** Summary slides.

---

## 19. Evaluation Rubric (100 Marks)

| Criteria | Marks | Details |
|---|---|---|
| **Working Features** | **20** | Image uploading works, CNN model classifies images, Gemini compiles recommendations, and logs save. |
| **Code Quality** | **15** | PEP 8 styling, cache configurations, and exception handlers. |
| **AI Implementation** | **15** | Tensor preprocessing logic, classifier inference performance, and advisor prompt validation. |
| **Documentation** | **10** | Detailed README.md, clean setup instructions, and PDF report. |
| **Git Usage** | **10** | Descriptive commits list, branch management, and repository hygiene. |
| **UI/UX Design** | **10** | Modern widgets layout, visual progress gauges, and clear metrics cards. |
| **Innovation & Extensions** | **10** | Webcam snapshots support, carbon offset trackers, or custom themes. |
| **Testing** | **5** | Pytest suites verifying the tensor reshaping and predictions. |
| **Presentation & Demo** | **5** | Video recording and slides explaining project value and findings. |
