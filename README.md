# DrugDiscoveryX

**DrugDiscoveryX** is an integrated platform that streamlines modern drug discovery by combining AI/ML-driven virtual screening, multi-source API data retrieval, and dynamic data visualization. With both CLI and GUI interfaces—including an interactive chat assistant powered by DialoGPT—DrugDiscoveryX simplifies target identification, lead optimization, and regulatory reporting for researchers and pharmaceutical professionals.

Repository: [https://github.com/TripathiNoSekai/DrugDiscoveryX](https://github.com/TripathiNoSekai/DrugDiscoveryX)

---

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Command-Line Interface (CLI)](#command-line-interface-cli)
  - [Graphical User Interface (GUI)](#graphical-user-interface-gui)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **AI/ML-Driven Virtual Screening:** Utilize machine learning (RandomForestRegressor) for predicting compound efficacy.
- **API Integrations:** Fetch data from PubChem, UniProt, ClinicalTrials.gov, DrugBank, and more.
- **Data Management & Visualization:** Export results to CSV/Excel and generate plots using Matplotlib and Seaborn.
- **User-Friendly Interfaces:** Choose between a CLI for streamlined operations or a GUI with an integrated chat assistant powered by DialoGPT.
- **Modular Design:** Easily extendable for future enhancements and additional functionalities.

---

## Prerequisites

- **Operating System:** Linux, macOS, or Windows
- **Python:** Version 3.7 or higher
- **Basic Knowledge:** Familiarity with Python and command-line operations

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/TripathiNoSekai/DrugDiscoveryX.git
   cd DrugDiscoveryX
Create and Activate a Virtual Environment (Recommended):

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
Install Dependencies:

Ensure you have a requirements.txt file in the repository. Then run:

bash
Copy
Edit
pip install -r requirements.txt
If tkinter is not installed with your Python distribution, install it via your system's package manager.

Configuration
Before running the script, set up the following environment variables:

bash
Copy
Edit
export ENTREZ_EMAIL="your_email@example.com"
export DRUG_DISCOVERY_DB="drug_discovery.db"
export VIRTUAL_SCREENING_MODEL="virtual_screening_model.pkl"
These variables configure your NCBI Entrez email, the path for the SQLite database, and the model file path.

Usage
Command-Line Interface (CLI)
Run specific pipeline functions using the following commands:

Initialize the Database:

bash
Copy
Edit
python DDP_4.py init-db
Train the Machine Learning Model:

bash
Copy
Edit
python DDP_4.py train
Perform Virtual Screening:

bash
Copy
Edit
python DDP_4.py virtual-screening <gene_name>
Fetch Gene Expression Data:

bash
Copy
Edit
python DDP_4.py fetch-gene <gene_name>
Other Commands:

Target Identification: python DDP_4.py target-id
Target Validation: python DDP_4.py target-validate <target_name>
Hit/Lead Identification: python DDP_4.py hit-lead <target_name>
Lead Optimization: python DDP_4.py lead-opt
ADMET Analysis: python DDP_4.py admet <compound_name>
Simulate Clinical Trial: python DDP_4.py simulate-clinical <compound_name>
Generate Regulatory Report: python DDP_4.py report
Export Compound Data: python DDP_4.py export-compound <compound_name>
Plot Molecular Weights: python DDP_4.py plot-weights <compound1> <compound2> ...
Fetch KEGG Pathway Data: python DDP_4.py kegg <gene_name>
Graphical User Interface (GUI)
To launch the GUI:

bash
Copy
Edit
python DDP_4.py
The GUI includes:

Pipeline Functions Tab: Execute pipeline commands with input fields and buttons.
Chat Assistant Tab: Interact with the DialoGPT-powered assistant for guidance.
About Tab: View developer information and clickable links.
Troubleshooting
Dependency Issues:
Ensure all required packages are installed in your virtual environment.

API Connection Errors:
Verify your network settings and ensure access to the external APIs.

Database Errors:
Check that the DRUG_DISCOVERY_DB environment variable is correctly set and that the file is accessible.

GUI Issues:
Confirm that Tkinter is properly installed for your Python distribution.

Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your enhancements or bug fixes. For major changes, please open an issue first to discuss what you would like to change.

License
This project is not licensed for open-source use. All rights are reserved by the author. If you wish to use or modify this code, please contact the author for permission.

Happy discovering!
