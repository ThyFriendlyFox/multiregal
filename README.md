# 🔮 MultiRegal - Automatic Multivariable Regression Analysis Calculator

An intelligent tool that automatically identifies the most important factors affecting your outcomes through AI-powered regression analysis.

## ✨ Features

- **🤖 AI-Powered Factor Identification**: Uses multiple statistical methods and LLM intelligence to identify the most important variables
- **📊 Interactive Web Interface**: Beautiful, user-friendly Streamlit interface for easy data analysis
- **🧹 Advanced Data Cleaning**: Automatically handles dirty data including missing values, duplicates, outliers, mixed data types, and inconsistent formatting
- **🔍 Comprehensive Analysis**: Performs correlation analysis, feature importance ranking, and regression modeling
- **📐 Formula Generation**: Creates both precise and simplified mathematical formulas
- **📈 Rich Visualizations**: Interactive charts, correlation heatmaps, and model comparisons
- **💡 Smart Insights**: LLM-generated recommendations and actionable insights
- **📋 Detailed Reports**: Complete cleaning and analysis reports with step-by-step documentation
- **💾 Export Capabilities**: Download results in JSON and CSV formats

## 🚀 Quick Start

### Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```

### Launch Web Interface

**Option 1: Use the launch script (Recommended)**
```bash
python run_app.py
```

**Option 2: Direct Streamlit command**
```bash
streamlit run app.py
```

The web interface will open at `http://localhost:8501`

## 📱 Web Interface Usage

### 1. **Upload Data**
- Upload a CSV file with your dataset
- Or use the demo data for quick testing
- Data preview and validation happens automatically

### 2. **Configure Analysis**
- Select your target variable (what you want to predict)
- Choose input features (or use all available)
- Set analysis parameters (max features, test size)

### 3. **Run Analysis**
- Click "🚀 Run Analysis" to start the process
- Watch real-time progress as the analysis runs
- Results appear in organized tabs

### 4. **Explore Results**
- **🎯 Key Findings**: Top factors and importance rankings
- **📊 Feature Analysis**: Correlation heatmaps and detailed analysis
- **🤖 Model Comparison**: Performance metrics across different models
- **📐 Formulas**: Generated mathematical formulas
- **💡 Insights**: AI-powered recommendations and insights
- **🧹 Data Cleaning Report**: Detailed report of all cleaning steps performed

### 5. **Export Results**
- Download comprehensive reports in JSON format
- Export factor rankings as CSV
- Save results for further analysis

## 🧹 Data Cleaning Capabilities

MultiRegal automatically handles messy, real-world data! It can clean:

- **🔄 Duplicate Records**: Automatically removes identical rows
- **💰 Currency & Number Formatting**: Handles "$1,234", "1,234.56", "€123", percentages
- **❌ Missing Values**: Smart imputation using mean/median based on data distribution
- **🔤 Mixed Data Types**: Converts "TRUE/FALSE", "Yes/No" to numeric, handles text in number columns
- **📊 Statistical Outliers**: Removes extreme values using IQR or Z-score methods
- **🏷️ Column Names**: Standardizes messy column names (spaces, special characters)
- **🔢 Data Type Issues**: Converts text representations to proper numeric formats
- **📏 Inconsistent Formats**: Handles spaces in numbers ("12 34" → 1234)

**Try the "Dirty Demo Data" option to see cleaning in action!**

## 📋 Data Requirements

- **Format**: CSV file
- **Minimum Size**: At least 10 rows (after cleaning)
- **Columns**: At least 2 numeric columns (after conversion)
- **Missing Data**: Up to 50% missing values (will be cleaned automatically)
- **Variable Types**: Any data types (will be converted to numeric where possible)

## 🛠️ Programmatic Usage

You can also use the analysis tools programmatically:

```python
from regression_analyzer.analysis_tools import (
    load_and_preprocess_data,
    identify_top_factors,
    perform_regression_analysis,
    generate_formula_and_insights
)
import pandas as pd

# Load your data
df = pd.read_csv("your_data.csv")

# Run analysis
top_factors = identify_top_factors(df, target_column="your_target")
model_results = perform_regression_analysis(df, "your_target", list(top_factors.keys()))
formulas = generate_formula_and_insights(df, "your_target", model_results)

print("Top factors:", top_factors)
print("Best model R²:", max([r['r2_score'] for r in model_results.values()]))
```

## 🎯 Use Cases

- **Business Analytics**: Identify factors affecting sales, revenue, or performance
- **Scientific Research**: Analyze experimental data and variable relationships
- **Financial Analysis**: Understand factors influencing financial metrics
- **Healthcare**: Analyze factors affecting patient outcomes
- **Marketing**: Determine which factors drive customer behavior
- **Operations**: Optimize processes by understanding key variables

## 🧠 How It Works

1. **Data Preprocessing**: Handles missing values, validates data structure
2. **Factor Identification**: Uses multiple methods:
   - Pearson correlation analysis
   - F-test statistical significance
   - Random Forest feature importance
   - Recursive feature elimination
3. **Model Training**: Tests multiple regression models:
   - Linear Regression
   - Ridge Regression (L2 regularization)
   - Lasso Regression (L1 regularization)
4. **Formula Generation**: Creates mathematical representations
5. **Insight Generation**: LLM-powered analysis and recommendations

## 📊 Example Output

**Top Factors Identified:**
```
1. square_footage    0.8234
2. location_score    0.7891
3. age_of_building  -0.6543
4. num_bedrooms      0.5321
5. amenities_score   0.4567
```

**Generated Formula:**
```
price = 245.67 + 156.23 * square_footage + 89.45 * location_score - 12.34 * age_of_building
```

**Key Insights:**
- Square footage has the strongest positive impact on price
- Location score is the second most important factor
- Building age negatively affects price
- Model achieves 89.4% accuracy (R² = 0.894)

## 🤖 Google ADK Integration

This tool integrates with Google's Agent Development Kit for enhanced LLM capabilities:

```bash
# To use with ADK web interface
adk serve

# To use with ADK CLI
adk run regression_analyzer
```

## 🔧 Development

### Project Structure
```
multiregal/
├── regression_analyzer/          # Core analysis package
│   ├── __init__.py
│   ├── agent.py                 # LLM agent integration
│   └── analysis_tools.py        # Statistical analysis functions
├── utils/                       # Web interface utilities
│   ├── __init__.py
│   └── interface_helpers.py     # Streamlit helper functions
├── app.py                       # Main Streamlit application
├── run_app.py                   # Launch script
├── demo.py                      # Command-line demo
├── test_installation.py         # Installation tester
├── sample_data.csv              # Demo dataset
└── requirements.txt             # Dependencies
```

### Running Tests
```bash
python test_installation.py
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

If you encounter any issues:
1. Check the data requirements section
2. Ensure all dependencies are installed correctly
3. Try the demo data first to verify the installation
4. Check the console/terminal for error messages

For additional help, please open an issue in the repository. 