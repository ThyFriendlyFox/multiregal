"""
Streamlit Web Interface for Automatic Multivariable Regression Analysis Calculator

This application provides an intuitive web interface for analyzing data with intelligent
factor identification and regression modeling using LLM-guided analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import time
from pathlib import Path

# Import our analysis tools
from regression_analyzer.analysis_tools import (
    load_and_preprocess_data,
    identify_top_factors,
    perform_regression_analysis,
    generate_formula_and_insights,
    create_analysis_summary
)

# Import interface helpers
from utils.interface_helpers import (
    display_data_preview,
    create_correlation_heatmap,
    create_feature_importance_chart,
    create_model_comparison_chart,
    display_formula_results,
    display_insights,
    create_download_report,
    validate_data_for_analysis,
    display_validation_results
)


def initialize_session_state():
    """Initialize session state variables."""
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None


def main():
    """Main application function."""
    st.set_page_config(
        page_title="MultuRegal - Automatic Regression Analysis",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    # Header
    st.title("🔮 MultuRegal")
    st.markdown("### Automatic Multivariable Regression Analysis Calculator")
    st.markdown("*Discover the most important factors affecting your outcomes with AI-powered analysis*")
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ Configuration")
        
        # Demo data option
        use_demo_data = st.checkbox("Use Demo Data (House Prices)", value=False)
        
        if use_demo_data:
            st.info("Using sample house price data for demonstration")
            # Load demo data
            try:
                demo_df = pd.read_csv("sample_data.csv")
                st.session_state.uploaded_data = demo_df
            except Exception as e:
                st.error(f"Error loading demo data: {e}")
                use_demo_data = False
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload section
        if not use_demo_data:
            st.header("📁 Upload Your Data")
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="Upload a CSV file with your data. The file should contain numeric variables for analysis."
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.uploaded_data = df
                    st.success(f"✅ File uploaded successfully! ({len(df)} rows, {len(df.columns)} columns)")
                except Exception as e:
                    st.error(f"Error reading file: {e}")
                    st.session_state.uploaded_data = None
        
        # Data preview and validation
        if st.session_state.uploaded_data is not None:
            df = st.session_state.uploaded_data
            
            # Display data preview
            with st.expander("📊 Data Preview", expanded=True):
                display_data_preview(df, max_rows=50)
            
            # Data validation
            st.header("🔍 Data Validation")
            issues, warnings = validate_data_for_analysis(df)
            can_proceed = display_validation_results(issues, warnings)
            
            if can_proceed:
                # Configuration section
                st.header("⚙️ Analysis Configuration")
                
                # Target variable selection
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 2:
                    target_variable = st.selectbox(
                        "Select Target Variable (what you want to predict)",
                        options=numeric_cols,
                        help="Choose the dependent variable you want to analyze"
                    )
                    
                    # Feature selection
                    available_features = [col for col in numeric_cols if col != target_variable]
                    selected_features = st.multiselect(
                        "Select Features (leave empty to use all)",
                        options=available_features,
                        default=available_features,
                        help="Choose specific features to include in the analysis"
                    )
                    
                    if not selected_features:
                        selected_features = available_features
                    
                    # Analysis options
                    st.subheader("Analysis Options")
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        max_features = st.slider(
                            "Maximum Features to Analyze",
                            min_value=3,
                            max_value=min(20, len(selected_features)),
                            value=min(10, len(selected_features)),
                            help="Limit the number of top features to focus on"
                        )
                    
                    with col_b:
                        test_size = st.slider(
                            "Test Set Size (%)",
                            min_value=10,
                            max_value=40,
                            value=20,
                            help="Percentage of data to use for testing"
                        ) / 100
                    
                    # Analysis button
                    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                        run_analysis(df, target_variable, selected_features, max_features, test_size)
                
                else:
                    st.error("Need at least 2 numeric columns for analysis.")
    
    with col2:
        # Instructions and tips
        st.header("💡 How to Use")
        st.markdown("""
        1. **Upload Data**: Upload a CSV file with your dataset
        2. **Select Target**: Choose what you want to predict
        3. **Choose Features**: Select input variables (optional)
        4. **Run Analysis**: Click the button to start analysis
        5. **Review Results**: Explore charts, formulas, and insights
        """)
        
        st.header("📋 Data Requirements")
        st.markdown("""
        - **Format**: CSV file
        - **Size**: At least 10 rows
        - **Columns**: At least 2 numeric columns
        - **Missing Data**: Less than 50% missing values
        """)
        
        if st.session_state.uploaded_data is not None:
            st.header("📊 Quick Stats")
            df_stats = st.session_state.uploaded_data
            st.metric("Rows", len(df_stats))
            st.metric("Columns", len(df_stats.columns))
            st.metric("Numeric Columns", len(df_stats.select_dtypes(include=[np.number]).columns))
    
    # Display results if analysis is complete
    if st.session_state.analysis_complete and st.session_state.analysis_results:
        display_analysis_results()


def run_analysis(df, target_variable, selected_features, max_features, test_size):
    """Run the complete regression analysis."""
    st.header("🔄 Analysis in Progress")
    
    # Create a progress container
    progress_container = st.container()
    
    with progress_container:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Load and preprocess data
            status_text.text("🔧 Loading and preprocessing data...")
            progress_bar.progress(0.1)
            
            # Prepare data for analysis - convert DataFrame to CSV string format
            analysis_df = df[selected_features + [target_variable]].copy()
            csv_string = analysis_df.to_csv(index=False)
            data_summary = load_and_preprocess_data(csv_string, target_column=target_variable)
            
            # Step 2: Identify top factors
            status_text.text("🔍 Identifying most important factors...")
            progress_bar.progress(0.3)
            time.sleep(0.5)  # Small delay for UX
            
            top_factors = identify_top_factors(
                data_summary, 
                max_factors=max_features
            )
            
            # Step 3: Perform regression analysis
            status_text.text("🤖 Training regression models...")
            progress_bar.progress(0.6)
            time.sleep(0.5)
            
            # Get the list of top factors from the analysis results
            selected_factors = top_factors.get("top_factors", [])[:max_features]
            model_results = perform_regression_analysis(
                data_summary, 
                selected_factors
            )
            
            # Step 4: Generate formulas and insights
            status_text.text("📐 Generating formulas and insights...")
            progress_bar.progress(0.8)
            time.sleep(0.5)
            
            formula_results = generate_formula_and_insights(model_results)
            
            # Step 5: Create summary
            status_text.text("📊 Creating analysis summary...")
            progress_bar.progress(0.9)
            
            analysis_summary = create_analysis_summary(
                data_summary, 
                top_factors, 
                model_results, 
                formula_results
            )
            
            # Complete
            progress_bar.progress(1.0)
            status_text.text("✅ Analysis complete!")
            
            # Store results
            st.session_state.analysis_results = {
                'data_summary': data_summary,
                'top_factors': top_factors,
                'model_results': model_results,
                'formulas': formula_results,
                'insights': analysis_summary,
                'target_variable': target_variable,
                'features_used': selected_features,
                'selected_factors': selected_factors
            }
            st.session_state.analysis_complete = True
            
            # Success message
            st.success("🎉 Analysis completed successfully! Scroll down to see results.")
            time.sleep(1)
            
            # Clear progress
            progress_container.empty()
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            progress_bar.empty()
            status_text.empty()


def display_analysis_results():
    """Display the complete analysis results."""
    st.header("📊 Analysis Results")
    
    results = st.session_state.analysis_results
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Key Findings", 
        "📊 Feature Importance", 
        "🤖 Model Comparison", 
        "📐 Formulas", 
        "💡 Insights & Recommendations"
    ])
    
    with tab1:
        st.subheader("🎯 Top Factors Identified")
        
        if results['top_factors'] and results['top_factors'].get('status') == 'success':
            top_factors_data = results['top_factors']
            factor_scores = top_factors_data.get('factor_scores', {})
            
            # Feature importance chart
            if factor_scores:
                importance_chart = create_feature_importance_chart(factor_scores)
                if importance_chart:
                    st.plotly_chart(importance_chart, use_container_width=True)
            
            # Top factors summary
            st.subheader("📋 Factor Rankings")
            if factor_scores:
                factors_df = pd.DataFrame.from_dict(
                    factor_scores, 
                    orient='index', 
                    columns=['Importance Score']
                )
                factors_df['Rank'] = range(1, len(factors_df) + 1)
                factors_df = factors_df[['Rank', 'Importance Score']].round(4)
                st.dataframe(factors_df, use_container_width=True)
            
            # Show selected factors
            selected_factors = top_factors_data.get('top_factors', [])
            if selected_factors:
                st.subheader("🏆 Selected Top Factors")
                for i, factor in enumerate(selected_factors[:10], 1):
                    score = factor_scores.get(factor, 0)
                    st.write(f"{i}. **{factor}** (Score: {score:.4f})")
        else:
            st.error("❌ Factor analysis failed or no results available")
    
    with tab2:
        st.subheader("📊 Feature Analysis")
        
        # Correlation analysis
        if st.session_state.uploaded_data is not None:
            target_var = results['target_variable']
            
            # Correlation heatmap
            corr_fig = create_correlation_heatmap(
                st.session_state.uploaded_data, 
                target_var
            )
            if corr_fig:
                st.plotly_chart(corr_fig, use_container_width=True)
    
    with tab3:
        st.subheader("🤖 Model Performance Comparison")
        
        if results['model_results'] and results['model_results'].get('status') == 'success':
            model_data = results['model_results']
            model_results_dict = model_data.get('model_results', {})
            
            # Model comparison chart
            if model_results_dict:
                comparison_chart = create_model_comparison_chart(model_results_dict)
                if comparison_chart:
                    st.plotly_chart(comparison_chart, use_container_width=True)
            
            # Show best model
            best_model = model_data.get('best_model', 'Unknown')
            st.subheader(f"🏆 Best Performing Model: {best_model.title()}")
            
            # Detailed model results
            st.subheader("📈 Detailed Model Metrics")
            model_details = []
            for model_name, metrics in model_results_dict.items():
                if 'error' not in metrics:
                    model_details.append({
                        'Model': model_name.title(),
                        'Test R² Score': round(metrics.get('test_r2', 0), 4),
                        'Train R² Score': round(metrics.get('train_r2', 0), 4),
                        'Test RMSE': round(metrics.get('test_rmse', 0), 4),
                        'Test MAE': round(metrics.get('test_mae', 0), 4),
                        'Features Used': len(model_data.get('selected_factors', []))
                    })
            
            if model_details:
                model_df = pd.DataFrame(model_details)
                st.dataframe(model_df, use_container_width=True)
            else:
                st.warning("No successful model results to display")
        else:
            st.error("❌ Model analysis failed or no results available")
    
    with tab4:
        st.subheader("📐 Generated Formulas")
        display_formula_results(results['formulas'])
    
    with tab5:
        st.subheader("💡 Analysis Insights")
        display_insights(results['insights'])
    
    # Download section
    st.header("💾 Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        # Create download report
        report_json = create_download_report(results)
        st.download_button(
            label="📥 Download Full Report (JSON)",
            data=report_json,
            file_name=f"regression_analysis_{results['target_variable']}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # Create CSV summary
        if results['top_factors'] and results['top_factors'].get('status') == 'success':
            factor_scores = results['top_factors'].get('factor_scores', {})
            if factor_scores:
                factors_csv = pd.DataFrame.from_dict(
                    factor_scores, 
                    orient='index', 
                    columns=['Importance Score']
                ).to_csv()
                
                st.download_button(
                    label="📊 Download Factor Rankings (CSV)",
                    data=factors_csv,
                    file_name=f"top_factors_{results['target_variable']}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    # Reset analysis button
    if st.button("🔄 Run New Analysis", use_container_width=True):
        st.session_state.analysis_complete = False
        st.session_state.analysis_results = None
        st.rerun()


if __name__ == "__main__":
    main() 