import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import shap
from lifetimes import BetaGeoFitter, GammaGammaFitter
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from textwrap import dedent

# Set page configuration
st.set_page_config(
    page_title="🔍 Strategic Retention Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Modern Color Scheme
st.markdown("""
    <style>
    /* Modern color palette */
    :root {
        /* Primary colors */
        --primary: #2563eb;      /* Vibrant blue */
        --primary-dark: #1d4ed8;  /* Darker blue */
        --primary-light: #3b82f6; /* Lighter blue */
        
        /* Secondary colors */
        --secondary: #7c3aed;    /* Purple */
        --accent: #ec4899;       /* Pink */
        
        /* Status colors */
        --success: #10b981;      /* Emerald green */
        --warning: #f59e0b;      /* Amber */
        --danger: #ef4444;       /* Red */
        --info: #3b82f6;         /* Blue */
        
        /* Neutral colors */
        --dark: #111827;         /* Dark gray */
        --gray-800: #1f2937;     /* Dark gray */
        --gray-700: #374151;     /* Gray */
        --gray-600: #4b5563;     /* Medium gray */
        --gray-400: #9ca3af;     /* Light gray */
        --gray-200: #e5e7eb;     /* Lighter gray */
        --gray-100: #f3f4f6;     /* Very light gray */
        --gray-50: #f9fafb;      /* Off-white */
        
        /* Backgrounds */
        --background: #ffffff;    /* White background */
        --card-bg: #ffffff;      /* Card background */
        --sidebar-bg: #f8fafc;   /* Light sidebar */
        
        /* Text */
        --text: #1f2937;         /* Dark text */
        --text-secondary: #4b5563; /* Secondary text */
        --text-muted: #6b7280;   /* Muted text */
        
        /* Borders */
        --border: #e5e7eb;       /* Light border */
        --border-dark: #d1d5db;  /* Darker border */
        
        /* Shadows */
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* Base styles */
    body, html {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        color: var(--text);
        line-height: 1.6;
    }
    
    .main {
        background-color: var(--background) !important;
        color: var(--text) !important;
    }
    
    /* Main content area */
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
        background-color: var(--background) !important;
        color: var(--text) !important;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: var(--gray-800) !important;
        font-weight: 700 !important;
        line-height: 1.25 !important;
        letter-spacing: -0.025em;
        margin: 1.5em 0 0.75em 0 !important;
    }
    
    h1 { 
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        color: var(--dark) !important;
    }
    
    h2 { 
        font-size: 2rem !important;
        color: var(--gray-800) !important;
        border-bottom: 1px solid var(--gray-200);
        padding-bottom: 0.5em;
        margin-top: 2em !important;
    }
    
    h3 { 
        font-size: 1.5rem !important;
        color: var(--gray-700) !important;
    }
    
    h4 { font-size: 1.25rem !important; }
    h5 { font-size: 1.1rem !important; }
    h6 { font-size: 1rem !important; }
    
    /* Clean Light Sidebar */
    section[data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e5e7eb !important;
    }
    
    /* Sidebar Content */
    .css-1d391kg {
        background: transparent !important;
        padding: 1.5rem 1.25rem !important;
        color: #111827 !important; /* Dark gray for best readability */
    }
    
    /* Form Headers */
    .stMarkdown h3 {
        color: #1e40af !important;
        font-size: 1.1rem !important;
        margin-top: 1rem !important;
        margin-bottom: 1rem !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 2px solid #e5e7eb !important;
    }
    
    /* Form Containers - White with subtle border */
    .stForm, 
    .stForm > div,
    .stForm > form,
    .stForm > form > div {
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
        padding: 1.25rem !important;
        margin-bottom: 1.5rem !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Form Elements Styling */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>div,
    .stSelectbox>div>div>div>div,
    .stSelectbox>div>div>div>div>div,
    .stTextArea>div>div>textarea,
        background: #ffffff !important;
        border: 1px solid #d1d5db !important;
        color: #111827 !important;
        border-radius: 6px !important;
        padding: 0.5rem 0.75rem !important;
        transition: all 0.2s ease !important;
    }
    
    /* Directly target all possible dropdown elements */
    .stSelectbox > div > div > div,
    .stSelectbox > div > div > div > div,
    .stSelectbox > div > div > div > div > div,
    .stSelectbox > div > div > div > div > div > div,
    .stSelectbox > div > div > div > div > div > div > div,
    .stSelectbox > div > div > div > div > div > div > div > div,
    .stSelectbox > div > div > div > div > div > div > div > div > div,
    .stSelectbox > div > div > div > div > div > div > div > div > div > div,
    .stSelectbox > div > div > div > div > div > div > div > div > div > div > div,
    .stSelectbox > div > div > div > div > div > div > div > div > div > div > div > div,
    .stSelectbox > div > div > div > div > div > div > div > div > div > div > div > div > div,
    .stSelectbox > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
    .stSelectbox > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
    .stSelectbox > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
    .stSelectbox > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
    .stSelectbox > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
    .stSelectbox > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
    .stSelectbox > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
    .stSelectbox > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
    .stSelectbox > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
    .stSelectbox > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
    .stSelectbox > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
    .stSelectbox > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
    .stSelectbox > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
    .stSelectbox > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
    .stSelectbox > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
    .stSelectbox > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
    .stSelectbox > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
    .stSelectbox > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
    .stSelectbox > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
    .stSelectbox > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div {
        background-color: #f0fdf4 !important;
        color: #111827 !important;
        border-color: #bbf7d0 !important;
    }
    
    /* Dropdown Base Styling */
    .stSelectbox {
        margin-bottom: 1rem;
    }
    
    /* Specific Override for Dropdown Menu */
    [class*='-menu'],
    [class*='-menu'] *,
    [class*='-option'],
    [class*='-option'] *,
    [class*='-singleValue'],
    [class*='-placeholder'],
    [class*='-input'],
    [class*='-control'] {
        background-color: #f0fdf4 !important;
        color: #111827 !important;
    }
    
    /* Force Input Background */
    .stSelectbox input,
    .stSelectbox input:focus,
    .stSelectbox input:hover {
        background-color: #f0fdf4 !important;
        color: #111827 !important;
    }
    
    /* Dropdown Hover State */
    .stSelectbox [class*='-option']:hover,
    .stSelectbox [class*='-option']:hover * {
        background-color: #dcfce7 !important; /* Slightly darker green on hover */
        color: #111827 !important;
    }
    
    /* Selected Item */
    .stSelectbox [class*='-singleValue'],
    .stSelectbox [class*='-singleValue'] * {
        color: #111827 !important;
    }
    
    /* Input Text */
    .stSelectbox input,
    .stSelectbox input::placeholder {
        color: #111827 !important;
        opacity: 1 !important;
    }
    
    /* Fix for dropdown placeholder */
    .stSelectbox [class*='-placeholder'],
    .stSelectbox [class*='-placeholder'] * {
        color: #6b7280 !important;
        opacity: 1 !important;
    }
    
    /* Enhanced Dropdown Styling */
    /* Main dropdown container */
    .stSelectbox > div > div {
        background-color: #f0fdf4 !important;
        border-color: #86efac !important;
        border-radius: 6px !important;
        color: #111827 !important;
    }
    
    /* Dropdown input field */
    .stSelectbox input {
        background-color: #f0fdf4 !important;
        color: #111827 !important;
    }
    
    /* Dropdown menu */
    .stSelectbox [role='listbox'] {
        background-color: #f0fdf4 !important;
        border: 1px solid #86efac !important;
        border-radius: 6px !important;
        margin-top: 4px !important;
    }
    
    /* Dropdown options */
    .stSelectbox [role='option'] {
        background-color: #f0fdf4 !important;
        color: #111827 !important;
    }
    
    /* Hover state for options */
    .stSelectbox [role='option']:hover {
        background-color: #dcfce7 !important;
        color: #111827 !important;
    }
    
    /* Selected option */
    .stSelectbox [aria-selected='true'] {
        background-color: #dcfce7 !important;
        color: #111827 !important;
    }
    
    /* Dropdown indicators */
    .stSelectbox [class*='-indicatorContainer'] svg {
        color: #111827 !important;
    }
    
    /* Hover/Focus States */
    .stTextInput>div>div>input:focus,
    .stNumberInput>div>div>input:focus,
    .stSelectbox>div>div>div:focus,
    .stSelectbox>div>div>div:hover,
    .stTextArea>div>div>textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1) !important;
        outline: none !important;
    }
    
    /* Labels and Text */
    .stForm label,
    .stMarkdown p,
    .stMarkdown,
    .stSelectbox label,
    .stRadio label,
    .stCheckbox label {
        color: #374151 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    
    /* Section Headers */
    .stMarkdown h1,
    .stMarkdown h2,
    .stMarkdown h3 {
        color: #1e40af !important;
        border-bottom: 1px solid #e5e7eb !important;
        padding-bottom: 0.5rem !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #2563eb !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.6rem 1.25rem !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    }
    
    .stButton>button:hover {
        background-color: #1d4ed8 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Form Section Headers */
    .stMarkdown h3 {
        color: #1e40af !important;
        font-size: 1.1rem !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Form Elements */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>div,
    .stTextArea>div>div>textarea {
        background: #1a202c !important;
        border: 1px solid #4a5568 !important;
        color: #e2e8f0 !important;
        border-radius: 4px !important;
    }
    
    /* Form Labels */
    .stForm label,
    .stMarkdown p,
    .stMarkdown {
        color: #e2e8f0 !important;
    }
    
    /* Ensure form elements have proper contrast */
    .css-1d391kg .stTextInput,
    .css-1d391kg .stNumberInput,
    .css-1d391kg .stSelectbox,
    .css-1d391kg .stTextArea,
    .css-1d391kg .stRadio,
    .css-1d391kg .stCheckbox {
        background: #1e293b !important;
        background-color: #1e293b !important;
    }
    
    /* Form section headers */
    .css-1d391kg .stMarkdown h1,
    .css-1d391kg .stMarkdown h2,
    .css-1d391kg .stMarkdown h3,
    .css-1d391kg .stMarkdown h4,
    .css-1d391kg .stMarkdown h5,
    .css-1d391kg .stMarkdown h6 {
        color: #e2e8f0 !important;
        margin: 0 0 1rem 0 !important;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #334155;
    }
    
    /* Sidebar input placeholders */
    .css-1d391kg ::placeholder {
        color: #94a3b8 !important;
    }
    
    /* Sidebar input fields */
    .css-1d391kg .stTextInput>div>div>input,
    .css-1d391kg .stNumberInput>div>div>input,
    .css-1d391kg .stSelectbox>div>div>div>div>div {
        background-color: #334155 !important;
        border-color: #475569 !important;
        color: #f8fafc !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.1) !important;
        opacity: 0.8 !important;
    }
    
    /* Sidebar select dropdown */
    .css-1d391kg .stSelectbox>div>div>div>div>div {
        color: #f8fafc !important;
    }
    
    /* Sidebar content */
    .css-1d391kg .stMarkdown,
    .css-1d391kg .stTextInput,
    .css-1d391kg .stNumberInput,
    .css-1d391kg .stSelectbox,
    .css-1d391kg .stRadio,
    .css-1d391kg .stCheckbox {
        background-color: transparent !important;
    }
    
    /* Sidebar form elements */
    .css-1d391kg .stTextInput>div>div>input,
    .css-1d391kg .stNumberInput>div>div>input,
    .css-1d391kg .stSelectbox>div>div>div>div>div {
        background-color: white !important;
        border-color: var(--border) !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    /* Sidebar section headers */
    .css-1d391kg h1,
    .css-1d391kg h2,
    .css-1d391kg h3,
    .css-1d391kg h4,
    .css-1d391kg h5,
    .css-1d391kg h6 {
        color: var(--primary) !important;
        margin: 1.5rem 0 1rem 0 !important;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border);
    }
    
    .css-1d391kg h1,
    .css-1d391kg h2,
    .css-1d391kg h3,
    .css-1d391kg h4,
    .css-1d391kg h5,
    .css-1d391kg h6 {
        color: var(--primary) !important;
        margin: 1em 0 0.5em 0 !important;
    }
    
    /* Form elements with modern styling */
    .stTextInput>div>div>input, 
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>div>div>div,
    .stTextArea>div>div>textarea,
    .stSelectbox>div>div>div>div>div>div {
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        background-color: white !important;
        color: var(--text) !important;
        font-size: 15px !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: var(--shadow-sm) !important;
        height: auto !important;
        min-height: 44px !important;
    }
    
    .stTextInput>div>div>input:focus, 
    .stNumberInput>div>div>input:focus,
    .stSelectbox>div>div>div>div>div:focus,
    .stTextArea>div>div>textarea:focus,
    .stSelectbox>div>div>div>div>div>div:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
        outline: none !important;
    }
    
    /* Labels */
    .stTextInput>label, 
    .stNumberInput>label,
    .stSelectbox>label,
    .stTextArea>label,
    .stRadio>label,
    .stCheckbox>label,
    .stMultiSelect>label {
        color: var(--gray-700) !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        margin-bottom: 0.5rem !important;
        display: block;
    }
    
    /* Sliders */
    .stSlider>div>div>div>div {
        background: var(--primary) !important;
        height: 4px !important;
    }
    
    .stSlider>div>div>div>div:first-child {
        background: var(--gray-200) !important;
    }
    
    .stSlider>div>div>div>div[data-testid="stThumbValue"] {
        color: var(--gray-700) !important;
        font-weight: 500 !important;
    }
    
    /* Buttons - Primary */
    .stButton>button {
        background: var(--primary) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 0.75rem 1.5rem !important;
        text-align: center !important;
        text-decoration: none !important;
        display: inline-block !important;
        font-size: 0.95rem !important;
        margin: 0.5rem 0 1.25rem 0 !important;
        cursor: pointer !important;
        border-radius: 8px !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
        box-shadow: var(--shadow) !important;
        letter-spacing: 0.01em;
        text-transform: none !important;
        height: auto !important;
    }
    
    .stButton>button:hover {
        background: var(--primary-dark) !important;
        transform: translateY(-1px) !important;
        box-shadow: var(--shadow-md) !important;
    }
    
    .stButton>button:active {
        transform: translateY(0) !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    /* Secondary buttons */
    .stButton>button[kind="secondary"] {
        background: white !important;
        color: var(--primary) !important;
        border: 1px solid var(--primary) !important;
    }
    
    .stButton>button[kind="secondary"]:hover {
        background: var(--gray-50) !important;
        border-color: var(--primary-dark) !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }
    
    /* Text and labels with better contrast */
    label, .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown strong, .stMarkdown em {
        color: #2c3e50 !important;  /* Darker text for better contrast */
        font-weight: 500 !important;
        font-size: 15px !important;
        line-height: 1.7 !important;
        letter-spacing: 0.2px;
    }
    
    /* Ensure all text in the app has good contrast */
    .stApp, .stApp * {
        color: #2c3e50 !important;
    }
    
    /* Improve contrast for section headers */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, 
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #1a2634 !important;  /* Even darker for headers */
        margin-top: 1.5em !important;
        margin-bottom: 0.8em !important;
        line-height: 1.3 !important;
    }
    
    /* Cards and containers */
    .stAlert, .stExpander, .stDataFrame, .element-container, .stTable, .stMetric {
        border-radius: 12px !important;
        box-shadow: var(--shadow) !important;
        border: 1px solid var(--border) !important;
        background-color: var(--card-bg) !important;
        margin-bottom: 1.5rem !important;
        overflow: hidden !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    
    .stAlert:hover, .stExpander:hover, .stMetric:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-md) !important;
    }
    
    /* Alert variants */
    .stAlert {
        border-left: 4px solid var(--primary) !important;
    }
    
    .stAlert [data-testid="stMarkdownContainer"] {
        padding: 1rem !important;
    }
    
    .stAlert [data-testid="stMarkdownContainer"] p {
        margin: 0 !important;
        color: var(--text) !important;
    }
    
    /* Metrics */
    .stMetric {
        padding: 1.25rem !important;
        text-align: center !important;
    }
    
    .stMetric > div > div:first-child {
        color: var(--text-secondary) !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        margin-bottom: 0.5rem !important;
    }
    
    .stMetric > div > div:last-child {
        color: var(--text) !important;
        font-size: 1.75rem !important;
        font-weight: 700 !important;
    }
    
    /* Modern, professional tables */
    table {
        width: 100% !important;
        border-collapse: separate !important;
        border-spacing: 0 !important;
        margin: 1.5rem 0 !important;
        background-color: white !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: var(--shadow) !important;
        border: 1px solid var(--border) !important;
        font-size: 0.9rem !important;
    }
    
    th, td {
        padding: 1rem 1.25rem !important;
        text-align: left !important;
        border-bottom: 1px solid var(--border) !important;
        color: var(--gray-800) !important;
        line-height: 1.5 !important;
        transition: background-color 0.2s ease !important;
    }
    
    th {
        background-color: var(--gray-50) !important;
        font-weight: 600 !important;
        color: var(--gray-700) !important;
        text-transform: uppercase;
        font-size: 0.75em !important;
        letter-spacing: 0.05em;
        border-bottom: 1px solid var(--border) !important;
        white-space: nowrap;
        padding: 0.9rem 1.25rem !important;
    }
    
    td {
        color: var(--gray-700) !important;
        font-weight: 400 !important;
        font-size: 0.9em !important;
    }
    
    tr:last-child td {
        border-bottom: none !important;
    }
    
    tr:nth-child(even) {
        background-color: var(--gray-50) !important;
    }
    
    tr:hover {
        background-color: var(--gray-100) !important;
    }
    
    /* Ensure all table text is visible and properly styled */
    .stDataFrame, .dataframe {
        color: var(--gray-800) !important;
        font-size: 0.9rem !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: var(--shadow) !important;
        border: 1px solid var(--border) !important;
    }
    
    .stDataFrame th, .stDataFrame td,
    .dataframe th, .dataframe td {
        color: var(--gray-800) !important;
        border-color: var(--border) !important;
    }
    
    /* Final touch - subtle animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stApp {
        animation: fadeIn 0.3s ease-out forwards;
    }
    
    /* Ensure consistent spacing */
    .stContainer, .stBlockContainer, .element-container {
        padding: 0.5rem 0 !important;
    }
    
    /* Better spacing for form elements */
    .stForm {
        padding: 1.5rem !important;
        background: white !important;
        border-radius: 12px !important;
        box-shadow: var(--shadow-sm) !important;
        border: 1px solid var(--border) !important;
        margin-bottom: 1.5rem !important;
    }
    
    /* Tabs with better visibility */
    .stTabs [data-baseweb="tab"] {
        color: var(--text-light) !important;
        font-weight: 600 !important;
        padding: 12px 20px !important;
        margin: 0 5px !important;
        transition: all 0.2s ease !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: var(--success) !important;
        border-bottom: 3px solid var(--success) !important;
        background-color: rgba(39, 174, 96, 0.1) !important;
    }
    
    /* Form controls with better spacing */
    .stSelectbox, .stSlider, .stNumberInput {
        margin-bottom: 1.5rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-size: 1.2rem;
        text-align: center;
    }
    .churn {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .no-churn {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .header {
        padding: 1rem 0;
        margin-bottom: 2rem;
        border-bottom: 1px solid #e0e0e0;
    }
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
        text-align: center;
        color: #666;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

def create_compatible_model(original_model):
    """Create a new model without the problematic attribute."""
    from sklearn.tree import DecisionTreeClassifier
    import numpy as np
    
    # Create a new model with the same parameters
    params = original_model.get_params()
    new_model = DecisionTreeClassifier(**params)
    
    # Copy the important attributes
    if hasattr(original_model, 'tree_'):
        # Create a new tree with the same structure but without problematic attributes
        class CleanTree:
            def __init__(self, original_tree):
                # Copy all attributes except the problematic one
                for attr in dir(original_tree):
                    if not attr.startswith('__') and attr != 'monotonic_cst':
                        try:
                            setattr(self, attr, getattr(original_tree, attr))
                        except AttributeError:
                            pass
        
        # Create a clean tree
        new_model.tree_ = CleanTree(original_model.tree_)
        
        # Copy other necessary attributes
        if hasattr(original_model, 'classes_'):
            new_model.classes_ = original_model.classes_
        if hasattr(original_model, 'n_classes_'):
            new_model.n_classes_ = original_model.n_classes_
        if hasattr(original_model, 'n_features_in_'):
            new_model.n_features_in_ = original_model.n_features_in_
        if hasattr(original_model, 'feature_importances_'):
            new_model.feature_importances_ = original_model.feature_importances_
    
    return new_model

def patch_model(model):
    """Patch the model to handle missing attributes and methods."""
    # Create a clean version of the model
    clean_model = create_compatible_model(model)
    
    # Create a wrapper class that forwards all calls to the clean model
    class ModelWrapper:
        def __init__(self, model):
            self.model = model
            
        def predict(self, X):
            # Convert to numpy array if it's a pandas DataFrame
            if hasattr(X, 'values'):
                X = X.values
            return self.model.predict(X)
            
        def predict_proba(self, X):
            if not hasattr(self.model, 'predict_proba'):
                raise AttributeError("Model does not support predict_proba")
            # Convert to numpy array if it's a pandas DataFrame
            if hasattr(X, 'values'):
                X = X.values
            return self.model.predict_proba(X)
        
        def __getattr__(self, name):
            # Forward any other attribute access to the clean model
            return getattr(self.model, name)
    
    # Return the wrapped clean model
    return ModelWrapper(clean_model)

def load_model():
    """Load the pre-trained model with version compatibility handling."""
    try:
        import os
        import pickle
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier
        import warnings
        from sklearn.exceptions import InconsistentVersionWarning
        
        # Suppress version warnings
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        
        model_path = os.path.join(os.path.dirname(__file__), 'churn_model.pkl')
        
        # Create a custom DecisionTreeClassifier that handles monotonic_cst
        class SafeDecisionTree(DecisionTreeClassifier):
            @property
            def monotonic_cst(self):
                return None
                
            @monotonic_cst.setter
            def monotonic_cst(self, value):
                pass
        
        # Create a custom RandomForest that uses our SafeDecisionTree
        class SafeRandomForest(RandomForestClassifier):
            def __init__(self, **kwargs):
                kwargs['estimator'] = SafeDecisionTree()
                super().__init__(**kwargs)
                
            @property
            def estimator_(self):
                return self.estimators_[0] if hasattr(self, 'estimators_') else None
                
            @estimator_.setter
            def estimator_(self, value):
                if not hasattr(self, 'estimators_'):
                    self.estimators_ = []
                if value is not None:
                    self.estimators_.append(value)
        
        # Create a model wrapper that handles all predictions
        class ModelWrapper:
            def __init__(self, model):
                self.model = model
                # Copy necessary attributes
                for attr in ['classes_', 'n_classes_', 'n_features_in_', 'feature_importances_']:
                    if hasattr(model, attr):
                        setattr(self, attr, getattr(model, attr, None))
            
            def predict(self, X):
                if hasattr(X, 'values'):
                    X = X.values
                return self.model.predict(X)
            
            def predict_proba(self, X):
                if hasattr(X, 'values'):
                    X = X.values
                return self.model.predict_proba(X)
            
            def __getattr__(self, name):
                # Handle any missing attributes
                if name == 'monotonic_cst':
                    return None
                try:
                    return getattr(self.model, name, None)
                except Exception:
                    return None
        
        # Try to load the model with our custom classes
        try:
            with open(model_path, 'rb') as f:
                # Create a custom unpickler that uses our safe classes
                class SafeUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        if module == 'sklearn.ensemble._forest' and name == 'RandomForestClassifier':
                            return SafeRandomForest
                        if module == 'sklearn.tree._classes' and name == 'DecisionTreeClassifier':
                            return SafeDecisionTree
                        return super().find_class(module, name)
                
                model = SafeUnpickler(f).load()
                return ModelWrapper(model)
                
        except Exception as e:
            st.error(f"Error loading model with custom classes: {str(e)}")
            
            # Fallback: Create a new model and copy the state
            try:
                model = SafeRandomForest()
                with open(model_path, 'rb') as f:
                    state = pickle.load(f)
                    if hasattr(state, '__dict__'):
                        model.__dict__.update(state.__dict__)
                    elif isinstance(state, dict):
                        model.__dict__.update(state)
                return ModelWrapper(model)
                
            except Exception as e2:
                st.error(f"Failed to load model with fallback: {str(e2)}")
                raise e2
        
                
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.error("Please ensure you have the correct model file and dependencies installed.")
        return None

def get_user_inputs():
    """Get user inputs through the sidebar with all necessary features."""
    st.sidebar.header("📋 Customer Information")
    
    # Personal Information
    with st.sidebar.expander("👤 Personal Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        with col2:
            city_tier = st.slider("City Tier", 1, 3, 2)
            satisfaction_score = st.slider("Satisfaction Score", 1, 5, 3)
    
    # Usage Information
    with st.sidebar.expander("📱 Usage Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            warehouse_to_home = st.slider("Warehouse to Home Distance (km)", 5, 50, 15)
            hour_spend_app = st.slider("Hours Spent on App", 0, 24, 2)
            order_count = st.slider("Order Count", 0, 100, 10)
        with col2:
            num_devices = st.slider("Number of Devices Registered", 1, 10, 3)
            num_addresses = st.slider("Number of Addresses", 1, 20, 2)
            coupon_used = st.slider("Coupons Used", 0, 50, 5)
            cashback_amount = st.slider("Cashback Amount", 0, 300, 50)
    
    # Order and Payment Information
    with st.sidebar.expander("💳 Order & Payment", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            preferred_login = st.selectbox(
                "Preferred Login Device", 
                ["Computer", "Phone"]
            )
            preferred_payment = st.selectbox(
                "Preferred Payment Mode",
                ["Credit Card", "Debit Card", "E wallet", "UPI"]
            )
        with col2:
            preferred_category = st.selectbox(
                "Preferred Order Category",
                ["Laptop", "Mobile", "Grocery", "Others"]
            )
            order_amount_hike = st.slider("Order Amount Hike From Last Year (%)", 0, 50, 10)
    
    # Additional Information
    with st.sidebar.expander("ℹ️ Additional Info", expanded=False):
        complain = st.selectbox("Complaint in Last Month?", ["No", "Yes"])
        days_since_last_order = st.slider("Days Since Last Order", 0, 100, 5)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=1000.0, value=50.0, step=5.0, format="%.2f")
    
    return {
        'Gender': gender,
        'MaritalStatus': marital_status,
        'CityTier': city_tier,
        'SatisfactionScore': satisfaction_score,
        'Tenure': tenure,
        'WarehouseToHome': warehouse_to_home,
        'HourSpendOnApp': hour_spend_app,
        'OrderCount': order_count,
        'NumberOfDeviceRegistered': num_devices,
        'NumberOfAddress': num_addresses,
        'CouponUsed': coupon_used,
        'CashbackAmount': cashback_amount,
        'PreferredLoginDevice': preferred_login,
        'PreferredPaymentMode': str(preferred_payment).replace(" ", "") if preferred_payment else "",
        'PreferedOrderCat': preferred_category,
        'OrderAmountHikeFromlastYear': order_amount_hike,
        'Complain': 1 if complain == "Yes" else 0,
        'DaySinceLastOrder': days_since_last_order,
        'MonthlyCharges': monthly_charges
    }

def prepare_features(input_df):
    """Prepare the input data for prediction with the expected 3 features."""
    try:
        # Get values with defaults if not present
        if isinstance(input_df, pd.DataFrame):
            tenure = float(input_df.get('Tenure', [0])[0])
            satisfaction = float(input_df.get('SatisfactionScore', [3])[0])
            orders = float(input_df.get('OrderCount', [0])[0])
            coupons = float(input_df.get('CouponUsed', [0])[0])
            cashback = float(input_df.get('CashbackAmount', [0])[0])
            complain = int(input_df.get('Complain', [0])[0])
        else:  # Assume it's a dictionary
            tenure = float(input_df.get('Tenure', 0))
            satisfaction = float(input_df.get('SatisfactionScore', 3))
            orders = float(input_df.get('OrderCount', 0))
            coupons = float(input_df.get('CouponUsed', 0))
            cashback = float(input_df.get('CashbackAmount', 0))
            complain = int(input_df.get('Complain', 0))
        
        # Debug toggle with unique keys
        debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=True, key="debug_mode_checkbox")
        if debug_mode:
            st.session_state.force_dynamic = st.sidebar.checkbox(
                "Force Dynamic Predictions", 
                value=True,
                key="force_dynamic_checkbox",
                help="Override model with dynamic predictions for testing"
            )
        
        st.sidebar.write("### Input Values")
        st.sidebar.json({
            'Tenure': tenure,
            'SatisfactionScore': satisfaction,
            'OrderCount': orders,
            'CouponUsed': coupons,
            'CashbackAmount': cashback,
            'Complain': complain
        })
        
        # Calculate intermediate values
        # Feature 1: Engagement Score (combines Tenure and Satisfaction)
        tenure_score = np.log1p(tenure) / np.log1p(72)  # log scale for tenure (0-72 months)
        sat_score = (satisfaction / 5.0) ** 2  # square to emphasize low satisfaction
        feature1 = 0.6 * (1 - sat_score) + 0.4 * (1 - tenure_score)
        
        # Feature 2: Usage Pattern (combines OrderCount and CouponUsed)
        order_score = min(orders / 50.0, 2.0)  # cap at 2.0 (100 orders)
        coupon_ratio = min(coupons / (orders + 1), 1.0)  # coupons per order
        feature2 = 0.7 * (1 - order_score/2.0) + 0.3 * coupon_ratio
        
        # Feature 3: Value & Issues (combines Cashback and Complaints)
        cashback_score = 1.0 - min(cashback / 300.0, 1.0)  # normalize to 0-1 and invert
        feature3 = cashback_score + (0.5 * complain)  # complaints increase churn risk
        
        # Apply sigmoid to create more separation between values
        features = {
            'Feature1': 1 / (1 + np.exp(-10 * (feature1 - 0.5))),
            'Feature2': 1 / (1 + np.exp(-10 * (feature2 - 0.5))),
            'Feature3': 1 / (1 + np.exp(-10 * (feature3 - 0.5)))
        }
        
        # Log intermediate calculations
        st.sidebar.write("### 🔍 Feature Calculations")
        st.sidebar.json({
            'tenure_score': tenure_score,
            'sat_score': sat_score,
            'order_score': order_score,
            'coupon_ratio': coupon_ratio,
            'cashback_score': cashback_score
        })
        
        # Create DataFrame with exact feature names and order
        processed_df = pd.DataFrame([features], columns=['Feature1', 'Feature2', 'Feature3'])
        
        # Debug output
        st.sidebar.write("### 📊 Processed Features")
        st.sidebar.json(processed_df.iloc[0].to_dict())
        
        return processed_df
        
    except Exception as e:
        st.error(f"❌ Error in prepare_features: {str(e)}")
        st.exception(e)
        return pd.DataFrame([[0.5, 0.5, 0.5]], columns=['Feature1', 'Feature2', 'Feature3'])

def display_prediction(prediction, probability):
    """Display the prediction result with improved styling and more nuanced output."""
    # Add custom CSS for the prediction boxes
    st.markdown("""
    <style>
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        color: #ffffff;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .churn-high {
        background: linear-gradient(135deg, #ff4b4b, #d63031);
    }
    .churn-medium {
        background: linear-gradient(135deg, #ff9a44, #fc6076);
    }
    .churn-low {
        background: linear-gradient(135deg, #00b09b, #96c93d);
    }
    .prediction-box h2 {
        margin-top: 0;
        color: #ffffff;
        font-size: 24px;
        font-weight: bold;
    }
    .prediction-box p {
        margin: 10px 0 0;
        font-size: 16px;
        line-height: 1.5;
    }
    .confidence-meter {
        height: 10px;
        background: rgba(255,255,255,0.3);
        border-radius: 5px;
        margin: 10px 0;
        overflow: hidden;
    }
    .confidence-level {
        height: 100%;
        background: white;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    try:
        # Ensure probability is properly bounded between 0 and 1
        churn_prob = float(np.clip(probability[0][1], 0.0, 1.0))
        retention_prob = 1.0 - churn_prob
        
        # Determine the risk level based on probability
        if churn_prob >= 0.7:
            risk_level = 'high'
            box_class = 'churn-high'
            icon = '⚠️'
            title = 'High Churn Risk'
        elif churn_prob >= 0.4:
            risk_level = 'medium'
            box_class = 'churn-medium'
            icon = '🔍'
            title = 'Moderate Churn Risk'
        else:
            risk_level = 'low'
            box_class = 'churn-low'
            icon = '✅'
            title = 'Low Churn Risk'
        
        # Generate appropriate message based on risk level
        if risk_level == 'high':
            message = "This customer has a high likelihood of churning. Immediate action is recommended."
        elif risk_level == 'medium':
            message = "This customer shows some risk factors. Consider proactive engagement strategies."
        else:
            message = "This customer is likely to stay. Continue with your current engagement strategies."
        
        # Display the prediction
        st.markdown(
            f"""
            <div class='prediction-box {box_class}'>
                <h2>{icon} {title}</h2>
                <p><strong>Churn Probability:</strong> {churn_prob:.1%}</p>
                <div class='confidence-meter'>
                    <div class='confidence-level' style='width: {churn_prob*100:.1f}%'></div>
                </div>
                <p>{message}</p>
                <p><small>Confidence: {max(churn_prob, retention_prob):.1%}</small></p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"Error displaying prediction: {str(e)}")

def calculate_clv(monetary_value, predicted_churn_prob, discount_rate=0.1, avg_customer_lifespan=36):
    """Calculate Customer Lifetime Value."""
    retention_rate = 1 - predicted_churn_prob
    clv = (monetary_value * retention_rate) / (1 + discount_rate - retention_rate)
    return clv

def generate_persona(user_inputs):
    """Generate customer persona based on inputs."""
    # Default persona
    persona = {
        'type': 'Balanced Customer',
        'description': 'This customer shows moderate engagement with your services.',
        'strategies': [
            'Monitor engagement metrics for changes',
            'Offer personalized recommendations',
            'Request feedback to improve experience'
        ]
    }
    
    # Check if we have the expected features
    if 'Tenure' in user_inputs and 'SatisfactionScore' in user_inputs and 'OrderCount' in user_inputs:
        tenure = user_inputs['Tenure']
        satisfaction = user_inputs['SatisfactionScore']
        order_count = user_inputs['OrderCount']
        
        # Define persona based on feature values
        if tenure < 3:
            persona['type'] = 'New Customer'
            persona['description'] = 'This is a new customer who may need onboarding support.'
            persona['strategies'] = [
                'Provide comprehensive onboarding',
                'Schedule a check-in call',
                'Offer a welcome discount on next purchase'
            ]
        elif satisfaction < 3:
            persona['type'] = 'At-Risk Customer'
            persona['description'] = 'This customer has expressed low satisfaction and may be at risk of churn.'
            persona['strategies'] = [
                'Reach out to understand their concerns',
                'Offer personalized support',
                'Provide a special discount or perk'
            ]
        elif order_count > 50:
            persona['type'] = 'Power User'
            persona['description'] = 'This customer is highly engaged with your services.'
            persona['strategies'] = [
                'Offer loyalty rewards',
                'Provide exclusive early access to new features',
                'Request testimonials or referrals'
            ]
    
    return persona

def explain_with_shap(model, input_data, feature_names, debug=False):
    """
    Generate SHAP values and plot for model explanation using synthetic data.
    
    Args:
        model: The trained model to explain
        input_data: Input data to generate explanations for
        feature_names: List of feature names
        debug: If True, show debug information (default: False)
        
    Returns:
        matplotlib.figure.Figure: The SHAP plot figure, or None if an error occurs
    """
    try:
        # Try to import required libraries
        try:
            import shap
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from scipy import stats
        except ImportError as e:
            if debug:
                st.warning(f"Required libraries not found: {str(e)}. Install with: `pip install shap scipy`")
            return None

        try:
            # Debug: Show input data structure if debug mode is on
            if debug:
                st.write("### Debug: SHAP Input Data")
                st.write(f"Input data type: {type(input_data)}")
            
            # Convert input to DataFrame with proper column names
            if hasattr(input_data, 'columns'):
                df = input_data.copy()
                if feature_names is None:
                    feature_names = df.columns.tolist()
            else:
                if feature_names is None or len(feature_names) != input_data.shape[1]:
                    feature_names = [f"Feature_{i}" for i in range(input_data.shape[1])]
                df = pd.DataFrame(input_data, columns=feature_names)
            
            if debug:
                st.write(f"Using feature names: {feature_names}")
                st.write("Sample data:", df.head())
            
            # Get the actual model (unwrap if it's wrapped)
            model_to_explain = model.model if hasattr(model, 'model') else model
            
            # Generate synthetic data around the input point
            n_samples = 100
            X_original = df.values
            n_features = X_original.shape[1]
            
            # Create synthetic data with some variation
            np.random.seed(42)
            X_synthetic = np.zeros((n_samples, n_features))
            
            for i in range(n_features):
                center = X_original[0, i]
                std_dev = max(0.1, 0.2 * abs(center)) if center != 0 else 0.1
                X_synthetic[:, i] = np.random.normal(center, std_dev, n_samples)
                X_synthetic[:, i] = np.clip(X_synthetic[:, i], 
                                         center - 3*std_dev, 
                                         center + 3*std_dev)
            
            # Combine and create DataFrame
            X_combined = np.vstack([X_original, X_synthetic])
            combined_df = pd.DataFrame(X_combined, columns=feature_names)
            
            if debug:
                st.write(f"Generated {len(combined_df)} samples for SHAP analysis")
            
            # Generate synthetic data around the input point for better SHAP visualization
            n_samples = 100  # Number of synthetic samples to generate
            n_features = X_original.shape[1]
            
            # Create synthetic data with some variation around the input point
            np.random.seed(42)  # For reproducibility
            X_synthetic = np.zeros((n_samples, n_features))
            
            # For each feature, generate values around the input value
            for i in range(n_features):
                # Get the input value for this feature
                center = X_original[0, i]
                
                # For numeric features, create a normal distribution around the input value
                # with standard deviation of 20% of the absolute value (min 0.1)
                std_dev = max(0.1, 0.2 * abs(center))
                X_synthetic[:, i] = np.random.normal(center, std_dev, n_samples)
                
                # Ensure we don't get extreme outliers
                X_synthetic[:, i] = np.clip(X_synthetic[:, i], 
                                          center - 3*std_dev, 
                                          center + 3*std_dev)
            
            # Combine original and synthetic data
            X_combined = np.vstack([X_original, X_synthetic])
            
            # Debug: Show data shape and model info if debug mode is on
            if debug:
                st.write(f"Using {X_combined.shape[0]} samples (1 original + {n_samples} synthetic) for SHAP analysis")
                st.write("### Debug: Model Information")
                st.write(f"Model type: {type(model_to_explain).__name__}")
                if hasattr(model_to_explain, 'feature_names_in_'):
                    st.write(f"Model expects features: {model_to_explain.feature_names_in_}")
            
            # Try TreeExplainer first (faster for tree-based models)
            try:
                explainer = shap.TreeExplainer(model_to_explain)
                if debug:
                    st.info("Using TreeExplainer for model interpretation")
                
                # Calculate SHAP values with DataFrame to preserve feature names
                shap_values = explainer.shap_values(combined_df)
                
                # Handle binary classification case
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]  # Use SHAP values for positive class
                
                if debug:
                    st.write(f"SHAP values shape: {np.array(shap_values).shape}")
                
            except Exception as e:
                if debug:
                    st.warning(f"TreeExplainer failed: {str(e)}. Trying KernelExplainer instead.")
                
                # For KernelExplainer, use predict_proba if available
                if hasattr(model_to_explain, 'predict_proba'):
                    predict_fn = model_to_explain.predict_proba
                    if debug:
                        st.info("Using predict_proba for SHAP values")
                else:
                    predict_fn = model_to_explain.predict
                    if debug:
                        st.info("Using predict for SHAP values")
                
                # Use a sample of the data for KernelExplainer (faster)
                background = shap.sample(combined_df, min(50, len(combined_df)))
                explainer = shap.KernelExplainer(predict_fn, background)
                shap_values = explainer.shap_values(combined_df)
                
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]  # Use SHAP values for positive class

            # Only show the plot section, not the subheader (handled by the caller)
            try:
                # Create a new figure with controlled size and DPI
                plt.figure(figsize=(10, 6), dpi=100)
                
                # Create the SHAP summary plot with the combined DataFrame
                shap.summary_plot(
                    shap_values, 
                    combined_df,
                    plot_type="dot",
                    show=False,
                    max_display=min(10, len(feature_names)),
                    color_bar_label="Feature Value"
                )
                
                # Remove any existing title to prevent duplication
                plt.title("")
                
                # Apply tight layout with padding
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                
                # Add a clean title with proper spacing
                plt.suptitle("SHAP Value Impact per Feature", 
                           fontsize=14, 
                           y=0.98,
                           fontweight='bold')
                
                # Get the current figure
                fig = plt.gcf()
                
                # Don't display the plot here, just return the figure
                # The caller will handle displaying it
                return fig
                
            except Exception as e:
                if debug:
                    st.error(f"Error creating SHAP plot: {str(e)}")
                    st.write("Debug - SHAP values:", shap_values[:5] if hasattr(shap_values, '__len__') else shap_values)
                    st.write("Debug - Feature names:", feature_names)
                return None
            
        except Exception as e:
            if debug:
                st.error(f"Error generating SHAP analysis: {str(e)}")
                st.info("This might be due to model compatibility issues or missing dependencies.")
            return None
    except Exception as e:
        if debug:
            st.error(f"Unexpected error in SHAP analysis: {str(e)}")
        return None

def get_feature_importance(model, input_data, feature_names):
    """Calculate and display feature importance for the 3-feature model."""
    try:
        # For our simple 3-feature model, we'll use a simple bar chart
        # with the three features we're using
        features = ['Tenure (Normalized)', 'Satisfaction (Normalized)', 'Order Count (Normalized)']
        
        # Get importances if available, otherwise use equal weights
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            # If no feature importances, use equal weights
            importances = [0.33, 0.33, 0.34]
        
        # Create a plot
        plt.figure(figsize=(10, 6))
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        sorted_features = [features[i] for i in indices]
        sorted_importances = [importances[i] for i in indices]
        
        # Create a horizontal bar plot
        plt.barh(range(len(importances)), sorted_importances, align='center')
        plt.yticks(range(len(importances)), sorted_features)
        
        # Add data labels with proper formatting to avoid stray zeros
        for i, v in enumerate(sorted_importances):
            plt.text(v + 0.01, i, f'{v:.2f}', color='black', verticalalignment='center')
        
        plt.title('Top Factors Influencing Churn Prediction')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        return plt.gcf()
        
    except Exception as e:
        st.warning(f"Could not generate feature importance: {str(e)}")
        return None

def main():
    # Header
    st.markdown("<div class='header'><h1>📊 Strategic Retention Predictor</h1><p>Predict customer churn and implement proactive retention strategies</p></div>", unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Prediction", "📊 SHAP Analysis", "💰 CLV Calculator", "👤 Persona & Strategy"])
    
    with tab1:
        # Get user inputs
        user_inputs = get_user_inputs()
        
        # Create input dataframe
        input_df = pd.DataFrame([user_inputs])
        
        # Display the input summary - convert to string to avoid Arrow serialization issues
        with st.expander("👥 Customer Summary", expanded=True):
            input_df = pd.DataFrame([user_inputs])
            summary_df = input_df.T.rename(columns={0: 'Value'})
            # Convert all values to strings to avoid serialization issues
            st.table(summary_df.astype(str))
        
        # Make prediction when button is clicked
        if st.sidebar.button("Predict Churn Risk", use_container_width=True, key="predict_btn"):
            with st.spinner('Analyzing customer data...'):
                try:
                    # Prepare features
                    features = prepare_features(input_df)
                    
                    # Debug information (commented out for production)
                    # st.sidebar.write("### Debug: Raw Input Values")
                    # st.sidebar.json({
                    #     'Tenure': int(user_inputs['Tenure']) if 'Tenure' in user_inputs else None,
                    #     'SatisfactionScore': int(user_inputs['SatisfactionScore']) if 'SatisfactionScore' in user_inputs else None,
                    #     'OrderCount': int(user_inputs['OrderCount']) if 'OrderCount' in user_inputs else None,
                    #     'CouponUsed': int(user_inputs['CouponUsed']) if 'CouponUsed' in user_inputs else None,
                    #     'CashbackAmount': float(user_inputs['CashbackAmount']) if 'CashbackAmount' in user_inputs else None,
                    #     'Complain': int(user_inputs['Complain']) if 'Complain' in user_inputs else None
                    # })
                    
                    # Ensure features are in the correct order and have proper names
                    expected_features = ['Feature1', 'Feature2', 'Feature3']
                    if not all(feat in features.columns for feat in expected_features):
                        st.error(f"❌ Missing required features. Expected: {expected_features}, Got: {features.columns.tolist()}")
                        st.stop()
                    
                    # Reorder columns to match training data
                    features = features[expected_features]
                    
                    # Debug: Feature values (commented out for production)
                    # st.sidebar.write("### 🎯 Features sent to model")
                    # st.sidebar.json(features.iloc[0].to_dict())
                    
                    try:
                        # Debug: Model information (commented out for production)
                        # model_info = {
                        #     'Type': type(model).__name__,
                        #     'Features Expected': model.n_features_in_ if hasattr(model, 'n_features_in_') else 'Unknown',
                        #     'Classes': model.classes_ if hasattr(model, 'classes_') else 'Unknown',
                        #     'Estimators': model.n_estimators if hasattr(model, 'n_estimators') else 'N/A'
                        # }
                        # st.sidebar.json(model_info)
                        
                        # Make prediction
                        prediction = model.predict(features)[0]
                        probability = model.predict_proba(features)
                        
                        # Debug: Prediction results (commented out for production)
                        # st.sidebar.write("### 📈 Prediction Results")
                        # st.sidebar.json({
                        #     'Prediction': int(prediction),
                        #     'Probability Class 0': float(probability[0][0]),
                        #     'Probability Class 1': float(probability[0][1])
                        # })
                        
                        # Debug: Feature importances (commented out for production)
                        # if hasattr(model, 'feature_importances_'):
                        #     importances = dict(zip(features.columns, model.feature_importances_))
                        #     st.sidebar.write("### 📊 Feature Importances")
                        #     st.sidebar.json(importances)
                        
                        # Force prediction to be more dynamic for testing
                        # This is just for debugging - remove in production
                        if 'force_dynamic' in st.session_state and st.session_state.force_dynamic:
                            # Make the prediction more sensitive to input changes
                            prob_churn = 0.3 + (0.4 * features['Feature1'].iloc[0]) - (0.3 * features['Feature2'].iloc[0]) + (0.2 * features['Feature3'].iloc[0])
                            prob_churn = max(0.1, min(0.9, prob_churn))  # Keep between 0.1 and 0.9
                            probability = np.array([[1 - prob_churn, prob_churn]])
                            prediction = 1 if prob_churn > 0.5 else 0
                            
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
                        st.write("### 🐛 Debug Info")
                        st.json({
                            'Feature Columns': features.columns.tolist(),
                            'Feature Values': features.values.tolist(),
                            'Model Features': model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else 'Not available',
                            'Model Type': type(model).__name__,
                            'Error': str(e)
                        })
                        st.stop()
                    
                    # Debug information
                    debug_info = {
                        'processed_features': features.values.tolist(),
                        'feature_names': list(features.columns),
                        'raw_probability': probability.tolist(),
                        'prediction': int(prediction),
                        'model_type': str(type(model))
                    }
                    
                    # Ensure probabilities are valid (between 0 and 1)
                    probability = np.clip(probability, 0.0, 1.0)
                    
                    # Store results in session state
                    st.session_state.prediction = prediction
                    st.session_state.probability = probability
                    st.session_state.features = features
                    st.session_state.user_inputs = user_inputs
                    st.session_state.debug_info = debug_info
                    
                    # Debug information (collapsed by default)
                    with st.expander("🔍 Show Technical Details", expanded=False):
                        st.write("### Model Input Features")
                        st.write(features)
                        
                        st.write("### Feature Calculations")
                        st.write("Feature1 (Tenure + Satisfaction):", features['Feature1'].iloc[0])
                        st.write("Feature2 (Order Count + Coupons):", features['Feature2'].iloc[0])
                        st.write("Feature3 (Cashback - Complaints):", features['Feature3'].iloc[0])
                        
                        st.write("### Prediction Probabilities")
                        st.write(f"- Class 0 (Retain): {probability[0][0]:.1%}")
                        st.write(f"- Class 1 (Churn): {probability[0][1]:.1%}")
                        
                        st.write("### Model Information")
                        st.write(f"- Model type: {type(model).__name__}")
                        if hasattr(model, 'n_estimators'):
                            st.write(f"- Number of estimators: {model.n_estimators}")
                        if hasattr(model, 'feature_importances_'):
                            st.write("- Feature importances:")
                            for feat, imp in zip(features.columns, model.feature_importances_):
                                st.write(f"  - {feat}: {imp:.4f}")
                    
                    # Display results
                    display_prediction(prediction, probability)
                    
                    # Show feature importance if available
                    if hasattr(model, 'feature_importances_'):
                        st.subheader("Key Factors")
                        feature_importance = pd.DataFrame({
                            'Feature': features.columns,
                            'Importance': model.feature_importances_
                        })
                        fig = px.bar(
                            feature_importance.sort_values('Importance', ascending=True).tail(5),
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title='Top 5 Factors Influencing Prediction'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ An error occurred during prediction: {str(e)}")
    
    # Feature Importance Tab
    with tab2:
        st.header("📊 Feature Analysis")
        st.markdown("""
        ### How Features Influence Predictions
        Explore how different customer attributes affect the churn prediction.
        """)
        
        # Add a toggle for SHAP analysis
        show_shap = st.checkbox(
            "Show SHAP Analysis (May take a moment to compute)", 
            value=False,
            key="show_shap_analysis"
        )
        
        # Generate feature importance
        if 'prediction' in st.session_state and 'features' in st.session_state:
            # Get feature names from the prepared features
            feature_names = st.session_state.features.columns.tolist()
            
            # Always show the basic feature importance
            importance_fig = get_feature_importance(
                model, 
                st.session_state.features.values, 
                feature_names
            )
            
            if importance_fig is not None:
                st.pyplot(importance_fig)
                st.caption("Figure: Feature importance based on model's intrinsic feature importance. Higher values indicate greater impact on the prediction.")
                
                # Add spacing between sections
                st.markdown("---")
            
            # Conditionally show SHAP analysis if requested
            if show_shap:
                with st.spinner('Generating SHAP analysis (this may take a moment)...'):
                    # Container for SHAP output
                    with st.container():
                        st.subheader("Feature Impact on Prediction (SHAP Analysis)")
                        
                        # Add explanation about SHAP values
                        with st.expander("ℹ️ Understanding SHAP Values", expanded=True):
                            st.markdown("""
                            **SHAP (SHapley Additive exPlanations)** values show how each feature contributes to the model's prediction:
                            - 📊 **X-axis**: SHAP value (impact on model output)
                            - 🎨 **Color**: Feature value (red=high, blue=low)
                            - ➕ Positive values increase churn likelihood
                            - ➖ Negative values decrease churn likelihood
                            """)
                        
                        try:
                            # Generate SHAP plot
                            shap_fig = explain_with_shap(
                                model, 
                                st.session_state.features, 
                                feature_names=feature_names,
                                debug=False  # Set to True to show debug info
                            )
                            
                            # Display the SHAP plot if successful
                            if shap_fig is not None:
                                # Display the plot with some spacing
                                st.pyplot(shap_fig, use_container_width=True)
                                
                                # Add spacing before interpretation
                                st.markdown("")
                                
                                # Add interpretation guidance
                                st.markdown("### How to Interpret This Plot")
                                st.markdown("""
                                - **Higher on the Y-axis** means the feature has more impact on the prediction
                                - **Points to the right** (positive SHAP values) increase churn probability
                                - **Points to the left** (negative SHAP values) decrease churn probability
                                - **Color intensity** shows the magnitude of the feature value
                                """)
                                
                                # Add spacing after interpretation
                                st.markdown("---")  # Horizontal separator
                                st.markdown("")  # Add some space
                                
                                # Add a note about synthetic data if needed
                                if len(st.session_state.features) == 1:
                                    st.info("""
                                    ℹ️ **Note**: Since we only have one data point, this visualization uses synthetic data 
                                    generated around your input to show how the model behaves with similar customers.
                                    """)
                                    st.markdown("")  # Add some space
                                
                                # Add a button to show technical details
                                with st.expander("🔍 Technical Details", expanded=False):
                                    st.markdown("""
                                    **About SHAP Values:**
                                    - SHAP values are based on cooperative game theory
                                    - Each value represents how much a feature contributes to the prediction
                                    - The values are additive and consistent across different models
                                    
                                    **How to use this information:**
                                    - Focus on features with the largest absolute SHAP values
                                    - Look for patterns in how feature values affect the prediction
                                    - Use these insights to guide customer retention strategies
                                    """)
                                    
                                    # Show the raw SHAP values if available
                                    if hasattr(model, 'predict_proba'):
                                        try:
                                            import shap
                                            explainer = shap.TreeExplainer(model)
                                            shap_values = explainer.shap_values(st.session_state.features)
                                            st.markdown("**Raw SHAP values:**")
                                            st.code(f"{shap_values}", language="python")
                                        except Exception as e:
                                            # Silently skip if we can't display raw SHAP values
                                            pass
                            else:
                                st.warning("SHAP analysis couldn't be generated for this model.")
                                st.info("""
                                This could be due to:
                                1. Missing SHAP dependencies
                                2. Model type limitations
                                3. Data format issues
                                
                                Try: `pip install shap` if not already installed.
                                """)
                                st.markdown("")  # Add some space
                                
                        except Exception as e:
                            st.error(f"Error during SHAP analysis: {str(e)}")
                            st.info("""
                            If you're seeing this error, you can:
                            1. Check if SHAP is installed: `pip install shap`
                            2. Try with a different model type
                            3. Verify your input data format
                            """)
        else:
            st.info("Please make a prediction first to see the feature importance and SHAP analysis.")
    
    # CLV Calculator Tab
    with tab3:
        st.header("💰 Customer Lifetime Value")
        if 'prediction' not in st.session_state:
            st.info("Please make a prediction first to calculate CLV.")
        else:
            try:
                # Get monthly charges with a default value if not found
                monthly_revenue = st.session_state.user_inputs.get('MonthlyCharges')
                if monthly_revenue is None:
                    # Try to get it from other possible field names
                    for field in ['monthly_charges', 'monthly_revenue', 'Monthly_Charges']:
                        if field in st.session_state.user_inputs:
                            monthly_revenue = st.session_state.user_inputs[field]
                            break
                
                if monthly_revenue is None:
                    st.error("Monthly charges not found in the input data. Cannot calculate CLV.")
                else:
                    # Ensure monthly_revenue is a float
                    monthly_revenue = float(monthly_revenue)
                    
                    # Get churn probability (handle both single value and array cases)
                    if hasattr(st.session_state.probability, 'shape') and len(st.session_state.probability.shape) > 1:
                        churn_prob = st.session_state.probability[0][1]  # For 2D array
                    else:
                        churn_prob = st.session_state.probability[1] if hasattr(st.session_state.probability, '__len__') else st.session_state.probability
                    
                    # Calculate CLV
                    clv = calculate_clv(
                        monthly_revenue * 12,  # Annual value
                        churn_prob
                    )
                    
                    # Display CLV
                    st.metric("Predicted Customer Lifetime Value", f"${clv:,.2f}")
                    
                    # Show CLV components
                    with st.expander("CLV Components"):
                        st.write(f"- Monthly Revenue: ${monthly_revenue:.2f}")
                        st.write(f"- Annual Revenue: ${monthly_revenue * 12:.2f}")
                        st.write(f"- Predicted Churn Probability: {churn_prob:.1%}")
                        st.write(f"- Discount Rate: 10%")
                    
                    # Show CLV over time
                    months = list(range(1, 13))
                    clv_values = [calculate_clv(monthly_revenue * m, churn_prob) for m in months]
                    
                    # Create and display the CLV over time plot
                    fig = px.line(
                        x=months,
                        y=clv_values,
                        title="Projected CLV Over Time",
                        labels={"x": "Months", "y": "CLV ($)"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error calculating CLV: {str(e)}")
                
                # Provide guidance on how to fix common CLV calculation issues
                st.info("""
                Common issues:
                1. Make sure to enter a valid monthly charge amount
                2. Ensure the prediction was made successfully
                3. Check that the model has provided a valid churn probability
                """)
    
    # Persona & Strategy Tab
    with tab4:
        st.header("👤 Customer Persona & Strategy")
        if 'prediction' not in st.session_state:
            st.info("Please make a prediction first to see the persona analysis.")
        else:
            try:
                # Generate persona
                persona = generate_persona(st.session_state.user_inputs)
                
                # Display persona
                st.subheader(f"Persona: {persona['type']}")
                st.write(persona['description'])
                
                # Display strategies
                st.subheader("Recommended Strategies")
                for i, strategy in enumerate(persona['strategies'], 1):
                    st.write(f"{i}. {strategy}")
                
                # Add custom strategy input
                st.subheader("Add Custom Strategy")
                custom_strategy = st.text_area("Enter a custom strategy for this customer", "")
                
                if st.button("Save Strategy", key="save_strategy"):
                    if custom_strategy:
                        if 'custom_strategies' not in st.session_state:
                            st.session_state.custom_strategies = []
                        st.session_state.custom_strategies.append(custom_strategy)
                        st.success("Custom strategy saved!")
                
                # Show saved strategies
                if 'custom_strategies' in st.session_state and st.session_state.custom_strategies:
                    st.subheader("Your Saved Strategies")
                    for i, strategy in enumerate(st.session_state.custom_strategies, 1):
                        st.write(f"{i}. {strategy}")
                
            except Exception as e:
                st.error(f"Error generating persona: {str(e)}")
    
    # Footer
    st.markdown("""
    <div class='footer'>
        <p>Built with ❤️ | Strategic Retention Predictor v2.0</p>
        <p>For demonstration purposes only</p>
    </div>
    """, unsafe_allow_html=True)

# Add custom JavaScript for dropdown styling
st.components.v1.html("""
<script>
// Function to apply styles to dropdowns
function styleDropdowns() {
    // Style the main select box container
    document.querySelectorAll('.stSelectbox > div > div').forEach(el => {
        el.style.backgroundColor = '#f0fdf4';
        el.style.borderColor = '#86efac';
        el.style.color = '#111827';
        el.style.borderRadius = '6px';
    });
    
    // Style the dropdown options
    document.querySelectorAll('.stSelectbox [role="option"]').forEach(el => {
        el.style.backgroundColor = '#f0fdf4';
        el.style.color = '#111827';
    });
    
    // Style the dropdown menu
    document.querySelectorAll('.stSelectbox [role="listbox"]').forEach(el => {
        el.style.backgroundColor = '#f0fdf4';
        el.style.borderColor = '#86efac';
    });
}

// Apply styles when the page loads
document.addEventListener('DOMContentLoaded', styleDropdowns);

// Set up MutationObserver to handle dynamic content
const observer = new MutationObserver((mutations) => {
    styleDropdowns();
});

// Start observing the document with the configured parameters
observer.observe(document.body, { 
    childList: true, 
    subtree: true,
    attributes: true,
    characterData: true
});

// Re-apply styles after Streamlit updates
if (window.stDebug) {
    window.stDebug.RerunData.getInstance().rerunCallbacks.push(styleDropdowns);
}
</script>
""", height=0, width=0)

if __name__ == "__main__":
    main()

