# 🧮 Automatic Multivariable Regression Analysis with Google ADK

An intelligent agent that automatically determines the factors with the greatest effect on an outcome by analyzing your data, running computations, and generating predictive formulas. Built with Google's Agent Development Kit (ADK).

## 🌟 Features

- **🤖 Intelligent Factor Identification**: Uses LLM reasoning combined with statistical methods to identify the most important factors
- **📊 Multiple Analysis Methods**: Employs correlation analysis, F-tests, Random Forest importance, and recursive feature elimination
- **🔬 Comprehensive Regression**: Tests Linear, Ridge, and Lasso regression models to find the best fit
- **📐 Formula Generation**: Produces both precise mathematical formulas and simplified versions for practical use
- **💡 Smart Insights**: Provides actionable insights about factor importance, statistical significance, and model performance
- **⚡ Easy to Use**: Just provide your data and target variable - the agent handles the rest

## 🏗️ Architecture

The application uses Google ADK's multi-tool architecture:

- **Main Agent**: LLM-powered orchestrator that guides the analysis process
- **Analysis Tools**: Specialized tools for data processing, factor analysis, and regression
- **Intelligent Decision Making**: The LLM decides which factors to include based on statistical evidence

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Google AI Studio API key or Google Cloud Project with Vertex AI enabled

### Installation

1. **Clone and install dependencies**:
```bash
pip install -r requirements.txt
pip install -e .
```

2. **Set up your Google AI credentials**:

**Option A: Google AI Studio (Recommended for development)**
```bash
export GOOGLE_GENAI_USE_VERTEXAI=FALSE
export GOOGLE_API_KEY=your_google_ai_studio_api_key_here
```

**Option B: Vertex AI (Recommended for production)**
```bash
export GOOGLE_GENAI_USE_VERTEXAI=TRUE
export GOOGLE_CLOUD_PROJECT=your_gcp_project_id
export GOOGLE_CLOUD_LOCATION=us-central1
# Also run: gcloud auth application-default login
```

### Run the Demo

```bash
python demo.py
```

This will analyze sample house price data and demonstrate the complete analysis workflow.

### Use with ADK Web UI

```bash
adk web
```

Then navigate to the provided URL and select the `regression_analyzer` agent.

### Use with ADK CLI

```bash
adk run .
```

## 📖 Usage Examples

### 1. Basic Data Analysis

```
Please analyze this dataset to identify factors affecting sales:

product_type,price,marketing_spend,season,competitor_count,sales
electronics,299,1000,summer,3,150
clothing,89,500,winter,5,80
electronics,199,800,spring,2,200
...

Target variable: sales
```

### 2. House Price Analysis

```
Analyze what factors most affect house prices in this dataset:

[CSV data with columns: sqft, bedrooms, location_score, age, price]

Target: price
```

### 3. Custom Analysis Request

```
I have customer data and want to predict satisfaction scores. 
Please identify the key factors and generate a prediction formula.

[Your CSV data]

Target variable: satisfaction_score
Please focus on factors that are actionable for business decisions.
```

## 🔧 How It Works

### 1. Data Preprocessing
- Loads CSV data (file or direct input)
- Handles missing values automatically
- Validates data structure and target variable

### 2. Intelligent Factor Analysis
The agent uses multiple methods to identify important factors:
- **Correlation Analysis**: Pearson correlation with target
- **F-Test Analysis**: Statistical significance testing
- **Random Forest**: Feature importance from ensemble model
- **Recursive Feature Elimination**: Systematic feature selection

### 3. Factor Selection
The LLM intelligently combines results from all methods to select the optimal factors, considering:
- Statistical significance
- Practical importance  
- Model complexity vs. explanatory power

### 4. Regression Analysis
- Tests multiple models (Linear, Ridge, Lasso)
- Performs train/test split for validation
- Generates comprehensive performance metrics

### 5. Formula Generation & Insights
- Creates precise mathematical formulas
- Provides simplified versions for practical use
- Generates actionable insights and recommendations

## 📊 Example Output

```
MULTIVARIABLE REGRESSION ANALYSIS SUMMARY
===============================================

## FINAL FORMULA
price ≈ 50000 + 150 * sqft + 8000 * neighborhood_score - 2000 * age

## KEY INSIGHTS
• Strong model performance: 87% of variance explained
• Most influential factor: sqft (coefficient: 149.23)
• Statistically significant factors (p<0.05): sqft, neighborhood_score, age
• Positive effects: sqft, neighborhood_score
• Negative effects: age

## RECOMMENDATIONS
• Focus on larger square footage and better neighborhoods for higher values
• Property age significantly reduces value - consider renovation potential
```

## 🛠️ Advanced Usage

### Custom Factor Selection

You can guide the agent's factor selection:

```
Please analyze this data but focus on factors that are:
1. Controllable by management
2. Have strong statistical significance (p < 0.01)
3. Show consistent effects across different time periods
```

### Multiple Target Analysis

```
Please analyze this dataset for two different outcomes:
1. Primary target: customer_satisfaction
2. Secondary target: repeat_purchase_likelihood

Compare which factors are important for each outcome.
```

## 📁 Project Structure

```
multiregal/
├── regression_analyzer/
│   ├── __init__.py              # Package initialization
│   ├── agent.py                 # Main ADK agent
│   └── analysis_tools.py        # Analysis and computation tools
├── sample_data.csv              # Example dataset
├── demo.py                      # Demonstration script
├── requirements.txt             # Python dependencies
├── pyproject.toml              # ADK configuration
└── README.md                   # This file
```

## 🧪 Available Tools

The agent has access to these specialized tools:

1. **`load_and_preprocess_data`**: Data loading, cleaning, and initial analysis
2. **`identify_top_factors`**: Multi-method factor importance analysis  
3. **`perform_regression_analysis`**: Comprehensive regression modeling
4. **`generate_formula_and_insights`**: Formula creation and interpretation
5. **`create_analysis_summary`**: Complete analysis documentation

## 🎯 Use Cases

- **Business Analytics**: Identify factors affecting sales, customer satisfaction, or operational efficiency
- **Real Estate**: Determine property value drivers for pricing models
- **Marketing**: Understand which campaign elements drive conversion
- **Product Development**: Analyze feature importance for user engagement
- **Financial Modeling**: Factor analysis for risk assessment or investment decisions
- **Research**: Academic or scientific factor analysis and hypothesis testing

## 🚀 Deployment

### Local Development
```bash
adk web  # Launch web UI
adk run  # CLI interface
```

### Production Deployment
The agent can be deployed to:
- **Google Cloud Run**: `adk deploy cloud_run`
- **Vertex AI Agent Engine**: `adk deploy agent_engine`
- **Custom Infrastructure**: Use standard ADK deployment patterns

## 🤝 Contributing

This application demonstrates ADK's capabilities for data analysis workflows. Feel free to:

- Add new analysis methods to `analysis_tools.py`
- Enhance the agent's instructions for specific domains
- Create specialized tools for particular data types
- Improve visualization and reporting features

## 📚 Learn More

- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [ADK Agents Guide](https://google.github.io/adk-docs/agents/)
- [ADK Tools Documentation](https://google.github.io/adk-docs/tools/)
- [Multi-Agent Systems](https://google.github.io/adk-docs/agents/multi-agent-systems/)

## 📄 License

This project is built using Google's Agent Development Kit and follows the same open-source principles. See individual component licenses for details.

---

**Ready to discover the hidden factors in your data? Just feed your dataset to the agent and watch it work its magic! 🎯📊** 