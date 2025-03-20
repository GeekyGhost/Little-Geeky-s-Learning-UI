# ui/styles.py

# CSS styles for the application
CSS = """
    /* Base Styles */
    .gradio-container { 
        background-color: #295095;
        font-family: 'Arial', sans-serif;
    }
    
    /* Header Styles */
    .header { 
        text-align: center; 
        padding: 20px;
        background-color: #295095;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    /* Achievement Card Styles */
    .achievement-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    
    .achievement-card:hover {
        transform: translateY(-2px);
    }
    
    /* Calculator Styles */
    .calculator {
        background-color: #2c3e50;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .calc-display {
        background-color: #34495e !important;
        color: #ffffff !important;
        font-size: 24px !important;
        text-align: right !important;
        margin-bottom: 10px !important;
        padding: 10px !important;
        border-radius: 5px !important;
    }
    .calc-btn {
        min-width: 50px !important;
        height: 50px !important;
        margin: 5px !important;
        font-size: 20px !important;
        border-radius: 5px !important;
        border: none !important;
        cursor: pointer !important;
    }
    .number-btn {
        background-color: #3498db !important;
        color: #ffffff !important;
    }
    .operator-btn {
        background-color: #e74c3c !important;
        color: #ffffff !important;
    }
    .function-btn {
        background-color: #f1c40f !important;
        color: #000000 !important;
    }
    .equals-btn {
        background-color: #2ecc71 !important;
        color: #ffffff !important;
    }
    
    /* Input/Output Areas */
    .reading-input, .math-input, .typing-input {
        border-radius: 8px !important;
        border: 2px solid #e2e8f0 !important;
    }
    
    /* Progress Display */
    .stats-display {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .achievement-card {
            margin: 5px;
        }
        
        .calc-btn {
            min-width: 40px !important;
            height: 40px !important;
            font-size: 16px !important;
        }
    }
"""