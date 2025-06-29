"""
Helper functions for the Streamlit interface of the regression analyzer.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import json


def display_data_preview(df, max_rows=100):
    """Display a preview of the uploaded data."""
    st.subheader("📊 Data Preview")
    
    # Basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Data types
    st.subheader("📋 Column Information")
    info_df = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.astype(str),
        'Non-Null Count': df.count(),
        'Missing Count': df.isnull().sum(),
        'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
    })
    st.dataframe(info_df, use_container_width=True)
    
    # Sample data
    st.subheader("🔍 Sample Data")
    display_rows = min(max_rows, len(df))
    st.dataframe(df.head(display_rows), use_container_width=True)
    
    if len(df) > max_rows:
        st.info(f"Showing first {max_rows} rows of {len(df)} total rows.")


def create_correlation_heatmap(df, target_col):
    """Create an interactive correlation heatmap."""
    # Calculate correlations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Feature Correlation Matrix"
    )
    
    # Highlight target variable correlations
    if target_col in corr_matrix.columns:
        target_corrs = corr_matrix[target_col].abs().sort_values(ascending=False)
        st.subheader(f"🎯 Correlations with {target_col}")
        
        # Create bar chart of correlations
        fig_bar = px.bar(
            x=target_corrs.index[1:],  # Exclude self-correlation
            y=target_corrs.values[1:],
            title=f"Feature Correlations with {target_col}",
            labels={'x': 'Features', 'y': 'Absolute Correlation'}
        )
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    return fig


def create_feature_importance_chart(importance_data):
    """Create a feature importance visualization."""
    if not importance_data:
        return None
    
    # Convert to DataFrame for easier plotting
    features = list(importance_data.keys())
    importances = list(importance_data.values())
    
    fig = px.bar(
        x=importances,
        y=features,
        orientation='h',
        title="Feature Importance Ranking",
        labels={'x': 'Importance Score', 'y': 'Features'},
        color=importances,
        color_continuous_scale="viridis"
    )
    fig.update_layout(height=max(400, len(features) * 30))
    
    return fig


def create_model_comparison_chart(model_results):
    """Create a comparison chart of different models."""
    if not model_results:
        return None
    
    models = list(model_results.keys())
    r2_scores = [results.get('test_r2', 0) for results in model_results.values() if 'error' not in results]
    rmse_scores = [results.get('test_rmse', 0) for results in model_results.values() if 'error' not in results]
    
    # Filter models to only include successful ones
    successful_models = [model for model, results in model_results.items() if 'error' not in results]
    
    if not successful_models:
        return None
    
    models = successful_models
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('R² Score (Higher is Better)', 'RMSE (Lower is Better)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # R² scores
    fig.add_trace(
        go.Bar(x=models, y=r2_scores, name="R² Score", marker_color="lightblue"),
        row=1, col=1
    )
    
    # RMSE scores
    fig.add_trace(
        go.Bar(x=models, y=rmse_scores, name="RMSE", marker_color="lightcoral"),
        row=1, col=2
    )
    
    fig.update_layout(title="Model Performance Comparison", showlegend=False)
    return fig


def display_formula_results(formula_results):
    """Display the generated formulas in a nice format."""
    if not formula_results:
        return
    
    st.subheader("📐 Generated Formulas")
    
    # Precise formula
    if 'precise_formula' in formula_results:
        st.subheader("🔬 Precise Formula")
        st.code(formula_results['precise_formula'], language='python')
    
    # Simplified formula
    if 'simplified_formula' in formula_results:
        st.subheader("✨ Simplified Formula")
        st.latex(formula_results['simplified_formula'])
    
    # Coefficients table
    if 'coefficients' in formula_results:
        st.subheader("📊 Coefficients")
        coef_df = pd.DataFrame.from_dict(
            formula_results['coefficients'], 
            orient='index', 
            columns=['Coefficient']
        )
        coef_df['Abs_Coefficient'] = abs(coef_df['Coefficient'])
        coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
        st.dataframe(coef_df, use_container_width=True)


def display_insights(insights):
    """Display the analysis insights in an organized manner."""
    if not insights:
        return
    
    st.subheader("🧠 Analysis Insights")
    
    # Key findings
    if 'key_findings' in insights:
        st.subheader("🔑 Key Findings")
        for finding in insights['key_findings']:
            st.write(f"• {finding}")
    
    # Recommendations
    if 'recommendations' in insights:
        st.subheader("💡 Recommendations")
        for rec in insights['recommendations']:
            st.write(f"• {rec}")
    
    # Statistical summary
    if 'statistical_summary' in insights:
        st.subheader("📈 Statistical Summary")
        for key, value in insights['statistical_summary'].items():
            st.write(f"**{key}**: {value}")


def create_download_report(analysis_results, filename="regression_analysis_report"):
    """Create a downloadable report of the analysis."""
    # Create a comprehensive report
    report = {
        "analysis_timestamp": pd.Timestamp.now().isoformat(),
        "data_summary": analysis_results.get('data_summary', {}),
        "top_factors": analysis_results.get('top_factors', {}),
        "model_results": analysis_results.get('model_results', {}),
        "formulas": analysis_results.get('formulas', {}),
        "insights": analysis_results.get('insights', {})
    }
    
    # Convert to JSON string
    report_json = json.dumps(report, indent=2, default=str)
    
    return report_json


def show_analysis_progress():
    """Show a progress bar and status updates during analysis."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    steps = [
        "Loading and preprocessing data...",
        "Identifying top factors...",
        "Training regression models...",
        "Generating formulas...",
        "Creating insights...",
        "Finalizing results..."
    ]
    
    for i, step in enumerate(steps):
        status_text.text(step)
        progress_bar.progress((i + 1) / len(steps))
        # In real implementation, you'd yield control back to the analysis functions
    
    status_text.text("Analysis complete!")
    return progress_bar, status_text


def validate_data_for_analysis(df):
    """Validate that the uploaded data is suitable for analysis."""
    issues = []
    warnings = []
    
    # Check minimum requirements
    if len(df) < 10:
        issues.append("Dataset has fewer than 10 rows. Need more data for reliable analysis.")
    
    if len(df.columns) < 2:
        issues.append("Dataset needs at least 2 columns (features + target).")
    
    # Check for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        issues.append("Need at least 2 numeric columns for regression analysis.")
    
    # Check missing values
    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    if missing_pct > 50:
        issues.append(f"Dataset has {missing_pct:.1f}% missing values. Too many missing values.")
    elif missing_pct > 20:
        warnings.append(f"Dataset has {missing_pct:.1f}% missing values. Consider data cleaning.")
    
    # Check for constant columns
    constant_cols = [col for col in numeric_cols if df[col].nunique() <= 1]
    if constant_cols:
        warnings.append(f"Columns with constant values detected: {constant_cols}")
    
    return issues, warnings


def display_validation_results(issues, warnings):
    """Display data validation results."""
    if issues:
        st.error("❌ **Data Issues Found:**")
        for issue in issues:
            st.error(f"• {issue}")
        return False
    
    if warnings:
        st.warning("⚠️ **Data Warnings:**")
        for warning in warnings:
            st.warning(f"• {warning}")
    
    if not issues and not warnings:
        st.success("✅ **Data looks good for analysis!**")
    
    return True  # Can proceed with analysis 