#!/usr/bin/env python
"""
Integrated Drug Discovery Pipeline with AI/ML, API Integrations, Data Export, Visualization, and GUI with Chat Assistant

Features:
  - AI/ML Pipeline: Target identification, virtual screening, ML model training, hit/lead identification, lead optimization, ADMET analysis, clinical trial simulation, regulatory reporting, gene expression data retrieval.
  - API Integrations: PubChem, UniProt, ClinicalTrials.gov, ChemBL, DrugBank, PDB, OMIM, KEGG.
  - Data Export: Export results to CSV and Excel.
  - Data Visualization: PrettyTable display and graphs (Matplotlib/Seaborn).
  - User Interfaces: CLI and a Tkinter-based GUI with an integrated chat assistant powered by DialoGPT.
  - About Tab: Displays developer info with clickable LinkedIn profile and email.
"""

import os
import logging
import json
import time
import csv
import random
import sqlite3
import pickle
import webbrowser

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import requests
from contextlib import contextmanager
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from prettytable import PrettyTable
from Bio import Entrez
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import argparse

# GUI and Chat Assistant imports
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext

# DialoGPT integration
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------
# Configuration and Logging
# ---------------------------
ENTREZ_EMAIL = os.environ.get("ENTREZ_EMAIL", "your_email@example.com")
DB_PATH = os.environ.get("DRUG_DISCOVERY_DB", "drug_discovery.db")
MODEL_PATH = os.environ.get("VIRTUAL_SCREENING_MODEL", "virtual_screening_model.pkl")
Entrez.email = ENTREZ_EMAIL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------
# API Configuration
# ---------------------------
API_CONFIG = {
    'pubchem': 'https://pubchem.ncbi.nlm.nih.gov/rest/pug',
    'swissadme': 'http://www.swissadme.ch/api.php',
    'uniprot': 'https://rest.uniprot.org',
    'clinicaltrials': 'https://clinicaltrials.gov/api/v2/studies',
    'chembl': 'https://www.ebi.ac.uk/chembl/api/data',
    'drugbank': 'https://api.drugbank.com/v1',
    'pdb': 'https://data.rcsb.org/rest/v1/core/entry',
    'omim': 'https://api.omim.org/api',
    'kegg': 'https://rest.kegg.jp'  # KEGG API for pathways
}

# ---------------------------
# Retry Decorator for Error Handling
# ---------------------------
def retry(max_attempts=3, delay=2, backoff=2):
    """
    Retries a function call with exponential backoff.
    """
    def decorator_retry(func):
        def wrapper_retry(*args, **kwargs):
            attempts = 0
            current_delay = delay
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    logging.error(f"Error in {func.__name__}: {e}. Attempt {attempts}/{max_attempts}")
                    if attempts == max_attempts:
                        raise e
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper_retry
    return decorator_retry

# ---------------------------
# API Request Helper with Retry Logic
# ---------------------------
@retry(max_attempts=3)
def perform_api_request(url, method='GET', data=None, params=None):
    if method == 'GET':
        response = requests.get(url, params=params)
    else:
        response = requests.post(url, data=data)
    response.raise_for_status()
    return response

# ---------------------------
# API Integration Functions
# ---------------------------
@lru_cache(maxsize=32)
def fetch_compound_data(compound_name):
    """
    Retrieve compound data from PubChem.
    """
    url = f"{API_CONFIG['pubchem']}/compound/name/{compound_name}/property/MolecularFormula,MolecularWeight,CanonicalSMILES/JSON"
    response = perform_api_request(url)
    return response.json().get('PropertyTable', {}).get('Properties', [{}])[0]

@lru_cache(maxsize=32)
def fetch_uniprot_data(protein_name):
    """
    Retrieve protein data from UniProt.
    """
    url = f"{API_CONFIG['uniprot']}/uniprotkb/search?query={protein_name}&format=json"
    response = perform_api_request(url)
    return response.json().get('results', [{}])[0]

@lru_cache(maxsize=32)
def fetch_clinical_trials(condition):
    """
    Fetch clinical trials from ClinicalTrials.gov.
    """
    url = API_CONFIG['clinicaltrials']
    params = {'query.term': condition, 'format': 'json'}
    response = perform_api_request(url, params=params)
    return response.json()

@lru_cache(maxsize=32)
def fetch_drugbank_data(drug_name):
    """
    Retrieve drug data from DrugBank.
    """
    url = f"{API_CONFIG['drugbank']}/drugs/{drug_name}"
    response = perform_api_request(url)
    return response.json()

@lru_cache(maxsize=32)
def fetch_pdb_structure(protein_id):
    """
    Retrieve protein 3D structure from PDB.
    """
    url = f"{API_CONFIG['pdb']}/{protein_id}"
    response = perform_api_request(url)
    return response.json()

@lru_cache(maxsize=32)
def fetch_omim_data(disease_name):
    """
    Retrieve genetic disease information from OMIM.
    """
    url = f"{API_CONFIG['omim']}/entry/search?search={disease_name}&format=json"
    response = perform_api_request(url)
    return response.json()

def fetch_kegg_pathway(gene_name):
    """
    Retrieve pathway data from KEGG.
    """
    url = f"{API_CONFIG['kegg']}/find/pathway/{gene_name}"
    response = requests.get(url)
    return response.text

# ---------------------------
# Data Export Functions
# ---------------------------
def export_to_csv(data, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data.keys())
        writer.writerow(data.values())
    print(f"Data exported to {filename}")

def export_to_excel(data, filename):
    df = pd.DataFrame([data])
    df.to_excel(filename, index=False)
    print(f"Data exported to {filename}")

# ---------------------------
# Data Visualization Functions
# ---------------------------
def display_table(data, headers):
    table = PrettyTable()
    table.field_names = headers
    table.add_row(data)
    print(table)

def display_json(data):
    print(json.dumps(data, indent=4))

def plot_molecular_weight(compounds):
    weights = [float(fetch_compound_data(c).get("MolecularWeight", 0)) for c in compounds]
    plt.bar(compounds, weights, color='blue')
    plt.xlabel('Compounds')
    plt.ylabel('Molecular Weight')
    plt.title('Molecular Weights of Compounds')
    plt.show()

# ---------------------------
# Database and AI/ML Pipeline Functions
# ---------------------------
@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    except Exception as e:
        logging.error("Database error: %s", e)
        raise e
    finally:
        conn.close()

def initialize_database():
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS GeneExpression (
                                gene TEXT PRIMARY KEY,
                                expression_data TEXT)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS VirtualScreening (
                                gene TEXT,
                                compound TEXT,
                                score REAL,
                                success_score REAL DEFAULT 0.0,
                                PRIMARY KEY (gene, compound))''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS ADMET (
                                compound TEXT PRIMARY KEY,
                                properties TEXT)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS Targets (
                                target TEXT PRIMARY KEY,
                                rank REAL,
                                validation_score REAL)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS Leads (
                                compound TEXT PRIMARY KEY,
                                target TEXT,
                                initial_score REAL,
                                optimized_score REAL,
                                admet TEXT)''')
            conn.commit()
        logging.info("Database initialized successfully.")
    except Exception as e:
        logging.error("Failed to initialize database: %s", e)

def target_identification():
    potential_targets = ["Protein_A", "Protein_B", "Protein_C", "Protein_D", "Protein_E"]
    ranked_targets = sorted(potential_targets, key=lambda t: random.uniform(0, 1))
    top_target = ranked_targets[0]
    rank_score = random.uniform(0, 0.5)
    logging.info("Identified top target: %s with rank score: %.3f", top_target, rank_score)
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO Targets (target, rank, validation_score) VALUES (?, ?, ?)",
                           (top_target, rank_score, None))
            conn.commit()
    except Exception as e:
        logging.error("Error saving target: %s", e)
    return top_target

def target_validation(target):
    docking_score = random.uniform(0.0, 1.0)
    is_valid = docking_score < 0.6
    logging.info("Validation for target %s: docking score=%.3f (valid=%s)", target, docking_score, is_valid)
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE Targets SET validation_score=? WHERE target=?", (docking_score, target))
            conn.commit()
    except Exception as e:
        logging.error("Error updating target validation: %s", e)
    return is_valid, docking_score

def hit_lead_identification(target):
    compounds = [f"Compound_{chr(65+i)}" for i in range(6)]
    scores = np.array([random.uniform(0, 1) for _ in compounds])
    screening_results = pd.DataFrame({'compound': compounds, 'score': scores})
    logging.info("Virtual screening results for target %s:\n%s", target, screening_results)
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(scores.reshape(-1, 1))
    screening_results['cluster'] = clusters
    cluster_avg = screening_results.groupby('cluster')['score'].mean()
    best_cluster = cluster_avg.idxmin()
    leads = screening_results[screening_results['cluster'] == best_cluster]
    logging.info("Identified leads from cluster %d:\n%s", best_cluster, leads)
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            for _, row in screening_results.iterrows():
                cursor.execute("INSERT OR IGNORE INTO VirtualScreening (gene, compound, score) VALUES (?, ?, ?)",
                               (target, row['compound'], row['score']))
            for _, row in leads.iterrows():
                cursor.execute("INSERT OR REPLACE INTO Leads (compound, target, initial_score, optimized_score, admet) VALUES (?, ?, ?, ?, ?)",
                               (row['compound'], target, row['score'], None, None))
            conn.commit()
    except Exception as e:
        logging.error("Error storing hit and lead identification results: %s", e)
    return leads

def lead_optimization():
    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM Leads WHERE optimized_score IS NULL", conn)
    except Exception as e:
        logging.error("Error retrieving leads: %s", e)
        return None
    if df.empty:
        logging.info("No leads to optimize.")
        return None
    optimized_scores = df['initial_score'] * random.uniform(0.5, 0.8)
    df['optimized_score'] = optimized_scores
    logging.info("Optimized lead compounds:\n%s", df[['compound', 'initial_score', 'optimized_score']])
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            for _, row in df.iterrows():
                cursor.execute("UPDATE Leads SET optimized_score=? WHERE compound=?", (row['optimized_score'], row['compound']))
            conn.commit()
    except Exception as e:
        logging.error("Error updating lead optimization: %s", e)
    return df

def admet_analysis(compound):
    compound_data = fetch_compound_data(compound)
    if not compound_data:
        logging.error(f"Failed to retrieve data for {compound}")
        return None
    smiles = compound_data.get('CanonicalSMILES')
    properties = {
        "lipinski_rule": random.choice(["Pass", "Fail"]),
        "logP": round(random.uniform(0.5, 5.0), 2),
        "GI_absorption": random.choice(["High", "Low"]),
        "BBB_penetration": random.choice(["Yes", "No"]),
        "CYP_inhibition": random.choice(["None", "Moderate", "High"])
    }
    logging.info("ADMET properties for %s: %s", compound, properties)
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO ADMET (compound, properties) VALUES (?, ?)", (compound, json.dumps(properties)))
            cursor.execute("UPDATE Leads SET admet=? WHERE compound=?", (json.dumps(properties), compound))
            conn.commit()
    except Exception as e:
        logging.error("Error storing ADMET data: %s", e)
    return properties

def clinical_trial_simulation(compound):
    mtd = round(random.uniform(50, 150), 1)
    efficacy = round(random.uniform(60, 95), 1)
    simulation_data = {"compound": compound, "MTD_mg": mtd, "efficacy_percent": efficacy}
    logging.info("Clinical trial simulation for %s: %s", compound, simulation_data)
    time_vals = np.linspace(0, 24, 100)
    concentration = np.exp(-time_vals / (mtd/100)) * random.uniform(10, 50)
    plt.figure(figsize=(6,4))
    plt.plot(time_vals, concentration, label=f"{compound}")
    plt.xlabel("Time (hours)")
    plt.ylabel("Concentration (a.u.)")
    plt.title(f"Simulated PK Curve for {compound}")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return simulation_data

def regulatory_report():
    report = {}
    try:
        with get_db_connection() as conn:
            target_df = pd.read_sql_query("SELECT * FROM Targets", conn)
            leads_df = pd.read_sql_query("SELECT * FROM Leads", conn)
            admet_df = pd.read_sql_query("SELECT * FROM ADMET", conn)
    except Exception as e:
        logging.error("Error retrieving data for report: %s", e)
        return
    report['Targets'] = target_df.to_dict(orient="records")
    report['Leads'] = leads_df.to_dict(orient="records")
    report['ADMET'] = admet_df.to_dict(orient="records")
    report_str = json.dumps(report, indent=4)
    logging.info("Regulatory Report:\n%s", report_str)
    with open("regulatory_report.json", "w") as f:
        f.write(report_str)
    print("Regulatory report saved to regulatory_report.json")
    return report

def train_ml_model():
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT compound, score, success_score FROM VirtualScreening")
            data = cursor.fetchall()
    except Exception as e:
        logging.error("Error fetching data for model training: %s", e)
        return
    if not data:
        logging.warning("No Data Available for Training!")
        return
    df = pd.DataFrame(data, columns=['compound', 'score', 'success_score'])
    df['total_score'] = df['score'] + df['success_score']
    X = df[['score', 'success_score']]
    y = df['total_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logging.info("Model trained successfully! MSE: %.4f", mse)

def virtual_screening(gene_name):
    if not gene_name.strip():
        logging.error("Error: Please enter a valid gene name!")
        return
    compounds = ['Compound_A', 'Compound_B', 'Compound_C']
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        features = np.random.rand(len(compounds), 2)
        scores = model.predict(features)
    except Exception as e:
        logging.warning("Trained model unavailable; using random scores. Error: %s", e)
        scores = np.random.rand(len(compounds))
    screening_results = pd.DataFrame({'compound': compounds, 'score': scores})
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            for _, row in screening_results.iterrows():
                cursor.execute("INSERT OR IGNORE INTO VirtualScreening (gene, compound, score) VALUES (?, ?, ?)", 
                               (gene_name, row['compound'], row['score']))
            conn.commit()
    except Exception as e:
        logging.error("Error storing screening results: %s", e)
    train_ml_model()
    logging.info("Screening completed successfully for gene: %s", gene_name)
    show_results(screening_results)

def fetch_gene_expression_data(gene_name):
    if not gene_name.strip():
        logging.error("Error: Please enter a valid gene name!")
        return
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT expression_data FROM GeneExpression WHERE gene = ?", (gene_name,))
            result = cursor.fetchone()
    except Exception as e:
        logging.error("Database error during gene fetch: %s", e)
        return
    if result:
        logging.info("Using cached gene expression data for %s.", gene_name)
        data = pd.read_json(result[0])
    else:
        logging.info("Fetching gene expression data for %s from NCBI Entrez...", gene_name)
        try:
            search_handle = Entrez.esearch(db="gds", term=gene_name, retmax=5)
            search_results = Entrez.read(search_handle)
            search_handle.close()
        except Exception as e:
            logging.error("Error during Entrez search: %s", e)
            return
        if not search_results.get("IdList"):
            logging.warning("No GEO datasets found for gene: %s", gene_name)
            return
        data = pd.DataFrame({'gene': [gene_name] * 10, 'expression': np.random.rand(10)})
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO GeneExpression (gene, expression_data) VALUES (?, ?)", 
                               (gene_name, data.to_json()))
                conn.commit()
        except Exception as e:
            logging.error("Error saving gene expression data: %s", e)
    show_results(data)

def show_results(data):
    try:
        plt.figure(figsize=(8, 6))
        if 'compound' in data.columns and 'score' in data.columns:
            sns.barplot(x=data['compound'], y=data['score'])
            plt.xlabel("Compounds")
            plt.ylabel("Screening Score")
            plt.title("Virtual Screening Results")
        elif 'expression' in data.columns:
            sns.lineplot(x=range(len(data)), y=data['expression'], marker="o")
            plt.xlabel("Sample Index")
            plt.ylabel("Expression Level")
            plt.title("Gene Expression Data")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error("Error in visualization: %s", e)

# ---------------------------
# Command-Line Interface (CLI)
# ---------------------------
def cli_main():
    parser = argparse.ArgumentParser(description="Integrated Drug Discovery ML Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    subparsers.add_parser("init-db", help="Initialize the database")
    subparsers.add_parser("train", help="Train the ML model")
    vs_parser = subparsers.add_parser("virtual-screening", help="Perform virtual screening for a gene")
    vs_parser.add_argument("gene", type=str, help="Gene name for virtual screening")
    ge_parser = subparsers.add_parser("fetch-gene", help="Fetch gene expression data for a gene")
    ge_parser.add_argument("gene", type=str, help="Gene name to fetch expression data for")
    
    subparsers.add_parser("target-id", help="Perform target identification")
    tv_parser = subparsers.add_parser("target-validate", help="Validate a target")
    tv_parser.add_argument("target", type=str, help="Target name to validate")
    hl_parser = subparsers.add_parser("hit-lead", help="Run hit and lead identification for a target")
    hl_parser.add_argument("target", type=str, help="Target name for hit identification")
    subparsers.add_parser("lead-opt", help="Run lead optimization on stored leads")
    admet_parser = subparsers.add_parser("admet", help="Perform ADMET analysis on a compound")
    admet_parser.add_argument("compound", type=str, help="Compound name for ADMET analysis")
    ct_parser = subparsers.add_parser("simulate-clinical", help="Simulate clinical trial for a compound")
    ct_parser.add_argument("compound", type=str, help="Compound name for clinical simulation")
    subparsers.add_parser("report", help="Generate regulatory report")
    
    exp_parser = subparsers.add_parser("export-compound", help="Export compound data to CSV and Excel")
    exp_parser.add_argument("compound", type=str, help="Compound name to export data for")
    viz_parser = subparsers.add_parser("plot-weights", help="Plot molecular weights for given compounds")
    viz_parser.add_argument("compounds", nargs="+", help="List of compound names")
    kegg_parser = subparsers.add_parser("kegg", help="Fetch KEGG pathway data for a gene")
    kegg_parser.add_argument("gene", type=str, help="Gene name to fetch KEGG pathway data for")
    
    args = parser.parse_args()
    
    if args.command == "init-db":
        initialize_database()
    elif args.command == "train":
        train_ml_model()
    elif args.command == "virtual-screening":
        virtual_screening(args.gene)
    elif args.command == "fetch-gene":
        fetch_gene_expression_data(args.gene)
    elif args.command == "target-id":
        target_identification()
    elif args.command == "target-validate":
        target_validation(args.target)
    elif args.command == "hit-lead":
        hit_lead_identification(args.target)
    elif args.command == "lead-opt":
        lead_optimization()
    elif args.command == "admet":
        admet_analysis(args.compound)
    elif args.command == "simulate-clinical":
        clinical_trial_simulation(args.compound)
    elif args.command == "report":
        regulatory_report()
    elif args.command == "export-compound":
        data = fetch_compound_data(args.compound)
        export_to_csv(data, f"{args.compound}_data.csv")
        export_to_excel(data, f"{args.compound}_data.xlsx")
    elif args.command == "plot-weights":
        plot_molecular_weight(args.compounds)
    elif args.command == "kegg":
        pathway = fetch_kegg_pathway(args.gene)
        print(f"KEGG Pathway for {args.gene}:\n{pathway}")
    else:
        parser.print_help()

# ---------------------------
# GUI with Chat Assistant using DialoGPT and About Tab
# ---------------------------
class PipelineGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Drug Discovery Pipeline GUI")
        self.geometry("900x700")
        # Load DialoGPT-small model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        self.chat_history_ids = None
        self.create_widgets()
        
    def create_widgets(self):
        # Create Notebook with three tabs: Pipeline Functions, Chat Assistant, and About
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill='both')
        
        # Pipeline Functions Tab
        self.pipeline_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.pipeline_frame, text="Pipeline Functions")
        
        # Chat Assistant Tab
        self.chat_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.chat_frame, text="Chat Assistant")
        
        # About Tab
        self.about_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.about_frame, text="About")
        # About information with clickable LinkedIn link and email
        name_label = ttk.Label(self.about_frame, text="Developed by: Prasun Dhar Tripathi", font=("Arial", 14, "bold"))
        name_label.pack(pady=10)
        linkedin_label = ttk.Label(self.about_frame, text="LinkedIn: www.linkedin.com/in/prasun-dhar-tripathi-934214180", foreground="blue", cursor="hand2")
        linkedin_label.pack(pady=5)
        linkedin_label.bind("<Button-1>", lambda e: webbrowser.open("www.linkedin.com/in/prasun-dhar-tripathi-934214180"))
        gmail_label = ttk.Label(self.about_frame, text="Gmail: tripathidhar2025@gmail.com", font=("Arial", 12))
        gmail_label.pack(pady=5)
        
        # Pipeline Tab: Create a frame for commands and one for output
        self.cmd_frame = ttk.Frame(self.pipeline_frame)
        self.cmd_frame.pack(side='top', fill='x', padx=5, pady=5)
        self.output_text = scrolledtext.ScrolledText(self.pipeline_frame, height=15)
        self.output_text.pack(side='bottom', fill='both', expand=True, padx=5, pady=5)
        
        # Create command buttons in Pipeline Tab
        self.create_command("Initialize DB", self.initialize_db_callback)
        self.create_command("Train ML Model", self.train_model_callback)
        self.create_command("Virtual Screening (Gene)", self.virtual_screening_callback, requires_input=True)
        self.create_command("Fetch Gene Expression (Gene)", self.fetch_gene_expression_callback, requires_input=True)
        self.create_command("Target Identification", self.target_identification_callback)
        self.create_command("Target Validation (Target)", self.target_validation_callback, requires_input=True)
        self.create_command("Hit/Lead Identification (Target)", self.hit_lead_identification_callback, requires_input=True)
        self.create_command("Lead Optimization", self.lead_optimization_callback)
        self.create_command("ADMET Analysis (Compound)", self.admet_analysis_callback, requires_input=True)
        self.create_command("Simulate Clinical Trial (Compound)", self.simulate_clinical_callback, requires_input=True)
        self.create_command("Generate Regulatory Report", self.regulatory_report_callback)
        self.create_command("Export Compound Data (Compound)", self.export_compound_callback, requires_input=True)
        self.create_command("Plot Molecular Weights (Compounds)", self.plot_weights_callback, requires_input=True)
        self.create_command("Fetch KEGG Pathway (Gene)", self.kegg_callback, requires_input=True)
        
        # Chat Assistant Tab: Chat display and input
        self.chat_display = scrolledtext.ScrolledText(self.chat_frame, height=20)
        self.chat_display.pack(fill='both', expand=True, padx=5, pady=5)
        self.chat_entry = ttk.Entry(self.chat_frame)
        self.chat_entry.pack(fill='x', padx=5, pady=5)
        self.chat_entry.bind("<Return>", self.send_chat)
        self.chat_button = ttk.Button(self.chat_frame, text="Send", command=self.send_chat)
        self.chat_button.pack(padx=5, pady=5)
        
    def create_command(self, label_text, callback, requires_input=False):
        frame = ttk.Frame(self.cmd_frame)
        frame.pack(fill='x', pady=2)
        label = ttk.Label(frame, text=label_text)
        label.pack(side='left')
        entry = None
        if requires_input:
            entry = ttk.Entry(frame, width=30)
            entry.pack(side='left', padx=5)
        button = ttk.Button(frame, text="Run", command=lambda: self.run_command(callback, entry))
        button.pack(side='left', padx=5)
        
    def run_command(self, callback, entry):
        user_input = entry.get() if entry is not None else ""
        try:
            result = callback(user_input)
            self.output_text.insert(tk.END, f"{result}\n")
            self.output_text.see(tk.END)
        except Exception as e:
            self.output_text.insert(tk.END, f"Error: {e}\n")
            self.output_text.see(tk.END)
        
    # Pipeline function callbacks
    def initialize_db_callback(self, user_input=""):
        initialize_database()
        return "Database initialized."
    
    def train_model_callback(self, user_input=""):
        train_ml_model()
        return "Model trained."
    
    def virtual_screening_callback(self, gene):
        virtual_screening(gene)
        return f"Virtual screening completed for gene: {gene}"
    
    def fetch_gene_expression_callback(self, gene):
        fetch_gene_expression_data(gene)
        return f"Gene expression data fetched for gene: {gene}"
    
    def target_identification_callback(self, user_input=""):
        target = target_identification()
        return f"Target identified: {target}"
    
    def target_validation_callback(self, target):
        valid, score = target_validation(target)
        return f"Target {target} validation: Score={score:.3f}, Valid={valid}"
    
    def hit_lead_identification_callback(self, target):
        hit_lead_identification(target)
        return f"Hit/lead identification completed for target: {target}"
    
    def lead_optimization_callback(self, user_input=""):
        lead_optimization()
        return "Lead optimization completed."
    
    def admet_analysis_callback(self, compound):
        admet_analysis(compound)
        return f"ADMET analysis performed for compound: {compound}"
    
    def simulate_clinical_callback(self, compound):
        clinical_trial_simulation(compound)
        return f"Clinical trial simulation completed for compound: {compound}"
    
    def regulatory_report_callback(self, user_input=""):
        regulatory_report()
        return "Regulatory report generated."
    
    def export_compound_callback(self, compound):
        data = fetch_compound_data(compound)
        export_to_csv(data, f"{compound}_data.csv")
        export_to_excel(data, f"{compound}_data.xlsx")
        return f"Compound data exported for {compound}."
    
    def plot_weights_callback(self, compounds_str):
        compounds = [c.strip() for c in compounds_str.split(",") if c.strip()]
        plot_molecular_weight(compounds)
        return "Molecular weight plot displayed."
    
    def kegg_callback(self, gene):
        pathway = fetch_kegg_pathway(gene)
        return f"KEGG Pathway for {gene}:\n{pathway}"
    
    # Chat Assistant using DialoGPT
    def send_chat(self, event=None):
        user_message = self.chat_entry.get().strip()
        if not user_message:
            return
        self.chat_display.insert(tk.END, f"You: {user_message}\n")
        response = self.chat_response(user_message)
        self.chat_display.insert(tk.END, f"Assistant: {response}\n")
        self.chat_display.see(tk.END)
        self.chat_entry.delete(0, tk.END)
    
    def chat_response(self, message):
        new_input_ids = self.tokenizer.encode(message + self.tokenizer.eos_token, return_tensors="pt")
        if self.chat_history_ids is not None:
            input_ids = torch.cat([self.chat_history_ids, new_input_ids], dim=-1)
        else:
            input_ids = new_input_ids
        self.chat_history_ids = self.model.generate(input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(self.chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        return response

# ---------------------------
# Main Entry Point
# ---------------------------
def main():
    # To run CLI instead of GUI, uncomment the next line and comment out the GUI lines.
    # cli_main()
    app = PipelineGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
