
# DrugDiscoveryX

**DrugDiscoveryX** is an integrated platform that streamlines modern drug discovery by combining **AI/ML-driven virtual screening**, **multi-source API data retrieval**, and **dynamic data visualization**. Featuring both **CLI** and **GUI** interfaces—including an **interactive chat assistant** powered by DialoGPT—DrugDiscoveryX simplifies **target identification**, **lead optimization**, and **regulatory reporting** for researchers and pharmaceutical professionals.


<img src="https://raw.githubusercontent.com/TripathiNoSekai/DrugDiscoveryX/main/DrugDiscoveryX.png" alt="DrugDiscoveryX" width="500">


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


![Drug Discovery X](https://github.com/TripathiNoSekai/DrugDiscoveryX/blob/main/Drug_Discovery_X.svg)



- **AI/ML-Driven Virtual Screening**  
  Employ a trained `RandomForestRegressor` to predict compound efficacy and streamline the screening workflow.

- **API Integrations**  
  Retrieve data from PubChem, UniProt, ClinicalTrials.gov, DrugBank, and more—ensuring you have the latest information on compounds and targets.

- **Data Management & Visualization**  
  Export results to CSV/Excel and generate insightful plots using Matplotlib and Seaborn. Present data clearly using PrettyTable.

- **User-Friendly Interfaces**  
  - **CLI** for straightforward, scriptable commands  
  - **GUI** built with Tkinter, featuring an integrated DialoGPT chat assistant for interactive guidance

- **Modular Design**  
  Easily extend the pipeline with additional functionalities, advanced ML models, or new API integrations.

---

## Prerequisites

- **Operating System:** Linux, macOS, or Windows  
- **Python:** Version 3.7 or higher  
- **Familiarity:** Basic knowledge of Python and command-line operations

> **Note:** If you plan on using the GUI, ensure `tkinter` is installed. Some operating systems may require installing it manually (e.g., `sudo apt-get install python3-tk` on Ubuntu).

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/TripathiNoSekai/DrugDiscoveryX.git
cd DrugDiscoveryX
```

### 2. Create & Activate a Virtual Environment (Recommended)

#### On macOS/Linux:
```bash
python -m venv venv
source venv/bin/activate
```

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn requests prettytable biopython scikit-learn transformers torch openpyxl tk

```

> **Note:**  
> - Ensure that `requirements.txt` is present in the repository.  
> - If `tkinter` is not installed:  
>   - **Ubuntu/Debian:** `sudo apt-get install python3-tk`  
>   - **macOS (Homebrew):** `brew install python-tk`  
>   - **Windows:** Typically included with standard Python distributions (verify if needed).

---

## Configuration

Before running the script, configure the following environment variables (adjust paths and email as needed):

```bash
export ENTREZ_EMAIL="your_email@example.com"
export DRUG_DISCOVERY_DB="drug_discovery.db"
export VIRTUAL_SCREENING_MODEL="virtual_screening_model.pkl"
```

- **ENTREZ_EMAIL:** Used by NCBI Entrez for identifying your requests.
- **DRUG_DISCOVERY_DB:** SQLite database file path.
- **VIRTUAL_SCREENING_MODEL:** File path to store/load the trained machine learning model.

> **Tip for Windows Users:**  
> Use `set` instead of `export` in Command Prompt, or adjust to PowerShell syntax.

---

## Usage

### Command-Line Interface (CLI)

Run specific pipeline tasks using the following commands:

- **Initialize the Database:**
  ```bash
  python DDP_4.py init-db
  ```
- **Train the Machine Learning Model:**
  ```bash
  python DDP_4.py train
  ```
- **Perform Virtual Screening:**
  ```bash
  python DDP_4.py virtual-screening <gene_name>
  ```
- **Fetch Gene Expression Data:**
  ```bash
  python DDP_4.py fetch-gene <gene_name>
  ```

#### Other Common Commands:
- **Target Identification:**
  ```bash
  python DDP_4.py target-id
  ```
- **Target Validation:**
  ```bash
  python DDP_4.py target-validate <target_name>
  ```
- **Hit/Lead Identification:**
  ```bash
  python DDP_4.py hit-lead <target_name>
  ```
- **Lead Optimization:**
  ```bash
  python DDP_4.py lead-opt
  ```
- **ADMET Analysis:**
  ```bash
  python DDP_4.py admet <compound_name>
  ```
- **Simulate Clinical Trial:**
  ```bash
  python DDP_4.py simulate-clinical <compound_name>
  ```
- **Generate Regulatory Report:**
  ```bash
  python DDP_4.py report
  ```
- **Export Compound Data:**
  ```bash
  python DDP_4.py export-compound <compound_name>
  ```
- **Plot Molecular Weights:**
  ```bash
  python DDP_4.py plot-weights <compound1> <compound2> ...
  ```
- **Fetch KEGG Pathway Data:**
  ```bash
  python DDP_4.py kegg <gene_name>
  ```

### Graphical User Interface (GUI)

To launch the GUI:

```bash
python DDP_4.py
```

The GUI provides:
- **Pipeline Functions Tab:** Execute core pipeline commands via buttons and input fields.
- **Chat Assistant Tab:** Interact with a DialoGPT-powered assistant for guidance.
- **About Tab:** View developer information and clickable links.

---

## Troubleshooting

- **Dependency Issues:**  
  Ensure you have installed all required packages (`pip install -r requirements.txt`) within your virtual environment.
- **API Connection Errors:**  
  Verify your network/firewall settings and ensure you can reach external APIs like PubChem or UniProt.
- **Database Errors:**  
  Confirm that `DRUG_DISCOVERY_DB` points to a valid file path and that you have the proper read/write permissions.
- **GUI Launch Problems:**  
  Ensure Tkinter is properly installed. If errors related to Tkinter occur, install the appropriate system package.
- **Logging:**  
  The script uses Python’s logging module. Check the console output for detailed error messages.

---

## Contributing

Contributions are welcome! To propose changes or enhancements:

1. **Fork the Project.**
2. **Create a New Feature Branch:**
   ```bash
   git checkout -b feature/new-feature
   ```
3. **Commit Your Changes:**
   ```bash
   git commit -m 'Add new feature'
   ```
4. **Push to Your Branch:**
   ```bash
   git push origin feature/new-feature
   ```
5. **Open a Pull Request on GitHub.**

> For major updates, please open an issue first to discuss your proposal.

---


 License (Apache 2.0)  

## License
This project is licensed under the **Apache License 2.0**.  
You are free to use, modify, and distribute this software as long as you comply with the license terms.

### **Key Terms**:
- 🔹 You **must** include this license in any copies or substantial portions of the software.
- 🔹 You **may** use this code for commercial and non-commercial projects.
- 🔹 **Patent Protection** is granted, ensuring contributors cannot sue for patent claims.
- 🔹 **Disclaimer**: This software is provided "as is," without warranties or guarantees.

Read the full **[LICENSE](LICENSE)** file for more details.
.
## About
Developed by Prasun Dhar Tripathi  
📩 Email: tripathidhar2025@gmail.com  
🔗 LinkedIn: [Click Here](https://www.linkedin.com/in/prasun-dhar-tripathi-934214180)

Happy Discovering!
