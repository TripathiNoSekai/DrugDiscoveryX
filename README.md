# DrugDiscoveryX

**DrugDiscoveryX** is an integrated platform that streamlines modern drug discovery by combining **AI/ML-driven virtual screening**, **multi-source API data retrieval**, and **dynamic data visualization**. With both **CLI** and **GUI** interfaces—including an **interactive chat assistant** powered by DialoGPT—DrugDiscoveryX simplifies **target identification**, **lead optimization**, and **regulatory reporting** for researchers and pharmaceutical professionals.

- **Repository:** [https://github.com/TripathiNoSekai/DrugDiscoveryX](https://github.com/TripathiNoSekai/DrugDiscoveryX)

---

## Table of Contents

1. [Features](#features)  
2. [Prerequisites](#prerequisites)  
3. [Installation](#installation)  
4. [Configuration](#configuration)  
5. [Usage](#usage)  
   - [Command-Line Interface (CLI)](#command-line-interface-cli)  
   - [Graphical User Interface (GUI)](#graphical-user-interface-gui)  
6. [Troubleshooting](#troubleshooting)  
7. [Contributing](#contributing)  
8. [License](#license)

---

## Features

- **AI/ML-Driven Virtual Screening**  
  Utilize machine learning (e.g., `RandomForestRegressor`) to predict compound efficacy and streamline the screening process.
- **API Integrations**  
  Fetch data from PubChem, UniProt, ClinicalTrials.gov, DrugBank, and more, ensuring up-to-date information for compounds and targets.
- **Data Management & Visualization**  
  Export results to CSV/Excel and generate plots using Matplotlib and Seaborn for clearer, data-driven insights.
- **User-Friendly Interfaces**  
  - **CLI** for quick, scriptable operations  
  - **GUI** with an integrated DialoGPT chat assistant for interactive guidance
- **Modular Design**  
  Easily extendable for future enhancements, additional functionalities, or integrations with other APIs.

---

## Prerequisites

- **Operating System:** Linux, macOS, or Windows  
- **Python:** Version 3.7 or higher  
- **Basic Knowledge:** Familiarity with Python and command-line operations

> **Note:** If you plan on using the GUI, make sure `tkinter` is installed on your system. Some operating systems or Python distributions may not include it by default.

---

## Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/TripathiNoSekai/DrugDiscoveryX.git
   cd DrugDiscoveryX
Create and Activate a Virtual Environment (Recommended)

bash
Copy
Edit
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
Ensure that requirements.txt is present in the repository.
If tkinter is not installed, install it via your system's package manager:
Ubuntu/Debian: sudo apt-get install python3-tk
macOS (Homebrew): brew install python-tk
Windows: Included with most standard Python distributions, but verify if needed.
Configuration
Before running the script, configure the following environment variables (adjust paths and email as needed):

bash
Copy
Edit
export ENTREZ_EMAIL="your_email@example.com"
export DRUG_DISCOVERY_DB="drug_discovery.db"
export VIRTUAL_SCREENING_MODEL="virtual_screening_model.pkl"
ENTREZ_EMAIL: Used by NCBI Entrez for identifying your requests.
DRUG_DISCOVERY_DB: SQLite database file path.
VIRTUAL_SCREENING_MODEL: File path to store/load the trained machine learning model.
Tip: On Windows, use set instead of export in Command Prompt, or powershell syntax in PowerShell.

Usage
Command-Line Interface (CLI)
Run specific pipeline functions via command-line commands:

Initialize the Database
bash
Copy
Edit
python DDP_4.py init-db
Train the Machine Learning Model
bash
Copy
Edit
python DDP_4.py train
Perform Virtual Screening
bash
Copy
Edit
python DDP_4.py virtual-screening <gene_name>
Fetch Gene Expression Data
bash
Copy
Edit
python DDP_4.py fetch-gene <gene_name>
Other Common Commands:
Target Identification:
bash
Copy
Edit
python DDP_4.py target-id
Target Validation:
bash
Copy
Edit
python DDP_4.py target-validate <target_name>
Hit/Lead Identification:
bash
Copy
Edit
python DDP_4.py hit-lead <target_name>
Lead Optimization:
bash
Copy
Edit
python DDP_4.py lead-opt
ADMET Analysis:
bash
Copy
Edit
python DDP_4.py admet <compound_name>
Simulate Clinical Trial:
bash
Copy
Edit
python DDP_4.py simulate-clinical <compound_name>
Generate Regulatory Report:
bash
Copy
Edit
python DDP_4.py report
Export Compound Data:
bash
Copy
Edit
python DDP_4.py export-compound <compound_name>
Plot Molecular Weights:
bash
Copy
Edit
python DDP_4.py plot-weights <compound1> <compound2> ...
Fetch KEGG Pathway Data:
bash
Copy
Edit
python DDP_4.py kegg <gene_name>
Graphical User Interface (GUI)
To launch the GUI:

bash
Copy
Edit
python DDP_4.py
The GUI provides:

Pipeline Functions Tab
Execute core pipeline commands via buttons and input fields.
Chat Assistant Tab
Interact with a DialoGPT-powered assistant for guidance on using the pipeline.
About Tab
View developer information and clickable links.
Troubleshooting
Dependency Issues
Make sure you have installed all required packages (pip install -r requirements.txt) within your virtual environment.
API Connection Errors
Check network/firewall settings and confirm you can reach external APIs like PubChem or UniProt.
Database Errors
Verify that DRUG_DISCOVERY_DB points to a valid file path and that you have the correct permissions to read/write.
GUI Launch Problems
Confirm that Tkinter is installed. If you see errors related to Tkinter, install the appropriate system package.
Logging: The script uses Python’s logging module. If you run into issues, check the console output for detailed error messages.

Contributing
Contributions are welcome! Please fork this repository and submit a pull request with your changes. For major updates, open an issue first to discuss the proposed modifications.

Fork the project
Create a new feature branch (git checkout -b feature/new-feature)
Commit your changes (git commit -m 'Add new feature')
Push to your branch (git push origin feature/new-feature)
Open a pull request on GitHub
License
All Rights Reserved. This project is not licensed for open-source use. If you wish to use or modify this code, please contact the author for permission.

