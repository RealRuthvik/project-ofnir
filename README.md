# **OFNIR AI Personal Trainer**

OFNIR is a high performance, professional AI fitness trainer wrapper built using **Gradio**, **LangGraph**, and **Groq**. It is designed to provide users with structured, biometric validated training and nutrition plans while maintaining a sleek, dark themed user interface.

## **🚀 Features**

* **Guided Plan Generation**: A step-by-step conversational flow for building comprehensive training and nutrition protocols.  
* **Biometric Validation**: Built-in logic to validate age, weight, height, and BMI to ensure user safety.  
* **Automated PDF Export**: Generates a professional PDF document of the final fitness plan using fpdf2.  
* **High-Speed Inference**: Powered by the llama-3.1-8b-instant model via the **Groq Cloud API** for near-instant responses.  
* **Premium Dark UI**: A customized, professional-grade obsidian interface with all Gradio branding and footers removed.  
* **Safety Guardrails**: Strict rules against providing advice on dangerous substances or unhealthy goals.

## **🛠️ Tech Stack**

* **Frontend**: [Gradio](https://gradio.app/) (Custom CSS)  
* **Orchestration**: [LangGraph](https://python.langchain.com/docs/langgraph/)  
* **LLM Framework**: [LangChain](https://python.langchain.com/docs/get_started/introduction)  
* **Inference**: [Groq Cloud API](https://groq.com/)  
* **PDF Generation**: [fpdf2](https://github.com/PyFPDF/fpdf2)

## **📋 Prerequisites**

* Python 3.9+  
* A Groq Cloud API Key

## **🔧 Installation**

1. **Clone the repository:**  
   git clone \[https://github.com/RealRuthvik/project-ofnir.git\](https://github.com/RealRuthvik/project-ofnir.git)  
   cd project-ofnir

2. **Install dependencies:**  
   pip install gradio langchain-groq langgraph fpdf2

3. Set up your API Key:  
   Replace "your\_key\_here" in the code with your actual Groq API key or set it as an environment variable:  
   export GROQ\_API\_KEY='your\_groq\_api\_key'

## **🖥️ Usage**

Run the application using Python:

python app.py

The interface will be accessible at http://127.0.0.1:7860.

## **🛡️ Safety Disclaimer**

OFNIR is an AI-driven tool provided for informational purposes. It is not a substitute for professional medical or nutritional advice. Users are encouraged to consult with a healthcare professional before starting any new fitness or diet program.

## **📄 License**

This project is licensed under the MIT License. See the LICENSE file for details.
