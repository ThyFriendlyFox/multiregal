"""
Analysis tools for multivariable regression analysis.
These tools handle data processing, factor identification, regression analysis, and formula generation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
from typing import Dict, List, Tuple, Any
import json
import io
import base64

# Try to import statsmodels, but make it optional
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("⚠️  statsmodels not available - detailed statistical analysis will be limited")


def load_and_preprocess_data(data_input: str, target_column: str) -> Dict[str, Any]:
    """
    Load data from CSV content or file path and preprocess it with comprehensive cleaning.
    
    Args:
        data_input: Either CSV file path or CSV content as string
        target_column: Name of the target/outcome column
    
    Returns:
        Dict containing preprocessed data information and cleaning report
    """
    try:
        # Import the data cleaner
        from .data_cleaner import clean_data_for_analysis
        
        # Clean the data using the comprehensive data cleaner
        df_clean, cleaning_report = clean_data_for_analysis(data_input, target_column, verbose=False)
        
        if cleaning_report["status"] != "success":
            return {
                "status": "error",
                "message": cleaning_report.get("message", "Data cleaning failed"),
                "cleaning_report": cleaning_report
            }
        
        # Validate we have a cleaned dataset
        if df_clean.empty:
            return {
                "status": "error",
                "message": "No data remaining after cleaning process"
            }
        
        # Get the cleaned target column name
        original_target = target_column
        target_column_clean = None
        
        # The cleaner maps column names, find the cleaned target column name
        if target_column.lower() in [col.lower() for col in df_clean.columns]:
            # Find exact match (case insensitive)
            for col in df_clean.columns:
                if col.lower() == target_column.lower():
                    target_column_clean = col
                    break
        
        if target_column_clean is None:
            return {
                "status": "error",
                "message": f"Target column '{original_target}' not found after cleaning. Available columns: {list(df_clean.columns)}"
            }
        
        # Validate target column exists after cleaning
        if target_column_clean not in df_clean.columns:
            return {
                "status": "error",
                "message": f"Target column '{target_column_clean}' not found after cleaning. Available columns: {list(df_clean.columns)}"
            }
        
        # Separate features and target
        y = df_clean[target_column_clean]
        
        # Only include numeric columns for analysis (exclude target and identifier columns)
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if target_column_clean in numeric_cols:
            numeric_cols.remove(target_column_clean)
        
        X = df_clean[numeric_cols]
        
        if X.empty:
            return {
                "status": "error",
                "message": "No numeric features available for analysis after cleaning"
            }
        
        # Calculate statistics
        try:
            target_stats = {
                "mean": float(y.mean()),
                "std": float(y.std()),
                "min": float(y.min()),
                "max": float(y.max()),
                "count": int(y.count())
            }
        except:
            target_stats = {"error": "Could not calculate target statistics"}
        
        try:
            feature_correlations = X.corr().to_dict()
        except:
            feature_correlations = {}
        
        try:
            target_correlations = X.corrwith(y).to_dict()
        except:
            target_correlations = {}
        
        # Prepare data info
        data_info = {
            "status": "success",
            "original_shape": cleaning_report.get("original_shape", df_clean.shape),
            "final_shape": df_clean.shape,
            "target_column": target_column_clean,
            "feature_columns": list(X.columns),
            "data_types": df_clean.dtypes.astype(str).to_dict(),
            "target_stats": target_stats,
            "feature_correlations": feature_correlations,
            "target_correlations": target_correlations,
            "cleaning_report": cleaning_report
        }
        
        # Store preprocessed data
        data_info["X_data"] = X.to_dict('records')
        data_info["y_data"] = y.tolist()
        
        return data_info
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error processing data: {str(e)}"
        }


def identify_top_factors(data_info: Dict[str, Any], max_factors: int = 10) -> Dict[str, Any]:
    """
    Use multiple techniques to identify the most important factors affecting the outcome.
    
    Args:
        data_info: Data information from load_and_preprocess_data
        max_factors: Maximum number of factors to identify
    
    Returns:
        Dict containing ranked factors and their importance scores
    """
    try:
        if data_info["status"] != "success":
            return data_info
        
        # Reconstruct data
        X = pd.DataFrame(data_info["X_data"])
        y = pd.Series(data_info["y_data"])
        
        if X.empty or y.empty:
            return {"status": "error", "message": "No data available for analysis"}
        
        factors_analysis = {}
        
        # Method 1: Correlation analysis
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        factors_analysis["correlation"] = correlations.to_dict()
        
        # Method 2: F-test feature selection
        try:
            f_scores, f_pvalues = f_regression(X, y)
            f_results = pd.Series(f_scores, index=X.columns).sort_values(ascending=False)
            factors_analysis["f_test"] = f_results.to_dict()
            factors_analysis["f_pvalues"] = dict(zip(X.columns, f_pvalues))
        except:
            factors_analysis["f_test"] = {}
            factors_analysis["f_pvalues"] = {}
        
        # Method 3: Random Forest feature importance
        try:
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
            factors_analysis["random_forest"] = rf_importance.to_dict()
        except:
            factors_analysis["random_forest"] = {}
        
        # Method 4: Recursive Feature Elimination
        try:
            lr = LinearRegression()
            rfe = RFE(lr, n_features_to_select=min(max_factors, len(X.columns)))
            rfe.fit(X, y)
            rfe_ranking = pd.Series(rfe.ranking_, index=X.columns).sort_values()
            factors_analysis["rfe_ranking"] = rfe_ranking.to_dict()
            factors_analysis["rfe_selected"] = [col for col, selected in zip(X.columns, rfe.support_) if selected]
        except:
            factors_analysis["rfe_ranking"] = {}
            factors_analysis["rfe_selected"] = []
        
        # Combine rankings to get top factors
        top_factors = []
        factor_scores = {}
        
        for factor in X.columns:
            score = 0
            # Correlation score (normalized)
            if factor in factors_analysis["correlation"]:
                score += factors_analysis["correlation"][factor] * 0.3
            
            # F-test score (normalized)
            if factor in factors_analysis["f_test"] and factors_analysis["f_test"]:
                max_f = max(factors_analysis["f_test"].values())
                if max_f > 0:
                    score += (factors_analysis["f_test"][factor] / max_f) * 0.3
            
            # Random Forest importance
            if factor in factors_analysis["random_forest"]:
                score += factors_analysis["random_forest"][factor] * 0.3
            
            # RFE bonus
            if factor in factors_analysis["rfe_selected"]:
                score += 0.1
            
            factor_scores[factor] = score
        
        # Sort factors by combined score
        sorted_factors = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)
        top_factors = [factor for factor, score in sorted_factors[:max_factors]]
        
        return {
            "status": "success",
            "top_factors": top_factors,
            "factor_scores": factor_scores,
            "detailed_analysis": factors_analysis,
            "selection_methods": {
                "correlation": "Pearson correlation with target variable",
                "f_test": "F-statistic for linear regression",
                "random_forest": "Feature importance from Random Forest",
                "rfe": "Recursive Feature Elimination ranking"
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error in factor identification: {str(e)}"
        }


def perform_regression_analysis(data_info: Dict[str, Any], selected_factors: List[str]) -> Dict[str, Any]:
    """
    Perform comprehensive regression analysis using the selected factors.
    
    Args:
        data_info: Data information from load_and_preprocess_data
        selected_factors: List of factor names to include in the analysis
    
    Returns:
        Dict containing regression results, coefficients, and model performance
    """
    try:
        if data_info["status"] != "success":
            return data_info
        
        # Reconstruct data
        X_full = pd.DataFrame(data_info["X_data"])
        y = pd.Series(data_info["y_data"])
        
        # Select only the specified factors
        available_factors = [f for f in selected_factors if f in X_full.columns]
        if not available_factors:
            return {"status": "error", "message": "No valid factors selected for analysis"}
        
        X = X_full[available_factors]
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        results = {}
        
        # Multiple regression models
        models = {
            "linear": LinearRegression(),
            "ridge": Ridge(alpha=1.0),
            "lasso": Lasso(alpha=0.1)
        }
        
        best_model = None
        best_score = -np.inf
        
        for model_name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Calculate metrics
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                train_mae = mean_absolute_error(y_train, y_pred_train)
                test_mae = mean_absolute_error(y_test, y_pred_test)
                
                # Store results
                results[model_name] = {
                    "coefficients": dict(zip(available_factors, model.coef_)),
                    "intercept": float(model.intercept_),
                    "train_r2": train_r2,
                    "test_r2": test_r2,
                    "train_rmse": train_rmse,
                    "test_rmse": test_rmse,
                    "train_mae": train_mae,
                    "test_mae": test_mae
                }
                
                # Track best model
                if test_r2 > best_score:
                    best_score = test_r2
                    best_model = model_name
                    
            except Exception as e:
                results[model_name] = {"error": str(e)}
        
        # Detailed statistical analysis using statsmodels (if available)
        if HAS_STATSMODELS:
            try:
                X_with_const = sm.add_constant(X)
                ols_model = sm.OLS(y, X_with_const).fit()
                
                statistical_summary = {
                    "r_squared": ols_model.rsquared,
                    "adj_r_squared": ols_model.rsquared_adj,
                    "f_statistic": ols_model.fvalue,
                    "f_pvalue": ols_model.f_pvalue,
                    "aic": ols_model.aic,
                    "bic": ols_model.bic,
                    "coefficients": dict(zip(["intercept"] + available_factors, ols_model.params)),
                    "std_errors": dict(zip(["intercept"] + available_factors, ols_model.bse)),
                    "t_values": dict(zip(["intercept"] + available_factors, ols_model.tvalues)),
                    "p_values": dict(zip(["intercept"] + available_factors, ols_model.pvalues)),
                    "confidence_intervals": dict(zip(["intercept"] + available_factors, 
                                                    ols_model.conf_int().values.tolist()))
                }
            except Exception as e:
                statistical_summary = {"error": f"Statistical analysis failed: {str(e)}"}
        else:
            # Fallback statistical summary using scikit-learn results
            statistical_summary = {
                "r_squared": best_score,
                "note": "Limited statistical analysis - statsmodels not available",
                "coefficients": dict(zip(["intercept"] + available_factors, 
                                       [results[best_model]["intercept"]] + 
                                       [results[best_model]["coefficients"][f] for f in available_factors]))
            }
        
        return {
            "status": "success",
            "selected_factors": available_factors,
            "model_results": results,
            "best_model": best_model,
            "statistical_summary": statistical_summary,
            "data_split": {
                "train_size": len(X_train),
                "test_size": len(X_test)
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error in regression analysis: {str(e)}"
        }


def generate_formula_and_insights(regression_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate mathematical formulas and insights from regression analysis results.
    
    Args:
        regression_results: Results from perform_regression_analysis
    
    Returns:
        Dict containing formulas, insights, and recommendations
    """
    try:
        if regression_results["status"] != "success":
            return regression_results
        
        best_model = regression_results["best_model"]
        model_results = regression_results["model_results"][best_model]
        statistical_summary = regression_results["statistical_summary"]
        
        # Generate formula
        coefficients = model_results["coefficients"]
        intercept = model_results["intercept"]
        
        formula_parts = [f"{intercept:.4f}"]
        
        for factor, coef in coefficients.items():
            sign = "+" if coef >= 0 else "-"
            formula_parts.append(f" {sign} {abs(coef):.4f} * {factor}")
        
        formula = "Outcome = " + "".join(formula_parts)
        
        # Generate simplified formula (rounded coefficients)
        simplified_parts = [f"{round(intercept, 2)}"]
        for factor, coef in coefficients.items():
            sign = "+" if coef >= 0 else "-"
            simplified_parts.append(f" {sign} {abs(round(coef, 2))} * {factor}")
        
        simplified_formula = "Outcome ≈ " + "".join(simplified_parts)
        
        # Generate insights
        insights = []
        
        # R-squared interpretation
        r2 = model_results["test_r2"]
        if r2 > 0.7:
            insights.append(f"Strong model performance: {r2:.1%} of variance explained")
        elif r2 > 0.5:
            insights.append(f"Moderate model performance: {r2:.1%} of variance explained")
        else:
            insights.append(f"Weak model performance: {r2:.1%} of variance explained")
        
        # Factor importance
        abs_coefficients = {k: abs(v) for k, v in coefficients.items()}
        most_important = max(abs_coefficients, key=abs_coefficients.get)
        insights.append(f"Most influential factor: {most_important} (coefficient: {coefficients[most_important]:.4f})")
        
        # Statistical significance (if available)
        if "p_values" in statistical_summary and isinstance(statistical_summary["p_values"], dict):
            significant_factors = []
            for factor, p_value in statistical_summary["p_values"].items():
                if factor != "intercept" and p_value < 0.05:
                    significant_factors.append(factor)
            
            if significant_factors:
                insights.append(f"Statistically significant factors (p<0.05): {', '.join(significant_factors)}")
            else:
                insights.append("No factors are statistically significant at p<0.05 level")
        
        # Effect direction insights
        positive_factors = [f for f, c in coefficients.items() if c > 0]
        negative_factors = [f for f, c in coefficients.items() if c < 0]
        
        if positive_factors:
            insights.append(f"Positive effects: {', '.join(positive_factors)}")
        if negative_factors:
            insights.append(f"Negative effects: {', '.join(negative_factors)}")
        
        # Model recommendations
        recommendations = []
        
        if r2 < 0.3:
            recommendations.append("Consider collecting more relevant features or checking data quality")
        if len(coefficients) > 10:
            recommendations.append("Consider reducing the number of factors for better interpretability")
        
        # Generate prediction function
        prediction_function = f"""
def predict_outcome({', '.join(coefficients.keys())}):
    return {intercept:.4f} + {' + '.join([f'{coef:.4f} * {factor}' for factor, coef in coefficients.items()])}
"""
        
        return {
            "status": "success",
            "formula": formula,
            "simplified_formula": simplified_formula,
            "prediction_function": prediction_function,
            "insights": insights,
            "recommendations": recommendations,
            "model_performance": {
                "r_squared": r2,
                "rmse": model_results["test_rmse"],
                "mae": model_results["test_mae"]
            },
            "factor_effects": {
                "positive_factors": positive_factors,
                "negative_factors": negative_factors,
                "most_important": most_important
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error generating formula and insights: {str(e)}"
        }


def create_analysis_summary(data_info: Dict[str, Any], factor_analysis: Dict[str, Any], 
                          regression_results: Dict[str, Any], formula_insights: Dict[str, Any]) -> str:
    """
    Create a comprehensive summary of the entire analysis process.
    
    Returns:
        Formatted string summary of the complete analysis
    """
    summary_parts = []
    
    summary_parts.append("# MULTIVARIABLE REGRESSION ANALYSIS SUMMARY")
    summary_parts.append("=" * 50)
    
    # Data overview
    if data_info["status"] == "success":
        summary_parts.append(f"\n## DATA OVERVIEW")
        summary_parts.append(f"Dataset shape: {data_info.get('final_shape', data_info.get('shape', 'Unknown'))}")
        summary_parts.append(f"Target variable: {data_info['target_column']}")
        summary_parts.append(f"Available features: {len(data_info['feature_columns'])}")
    
    # Factor selection
    if factor_analysis["status"] == "success":
        summary_parts.append(f"\n## FACTOR SELECTION")
        summary_parts.append(f"Top factors identified: {', '.join(factor_analysis['top_factors'][:5])}")
        summary_parts.append("Selection methods used:")
        for method, description in factor_analysis['selection_methods'].items():
            summary_parts.append(f"  - {method}: {description}")
    
    # Regression results
    if regression_results["status"] == "success":
        summary_parts.append(f"\n## REGRESSION ANALYSIS")
        summary_parts.append(f"Best performing model: {regression_results['best_model']}")
        summary_parts.append(f"Factors included: {', '.join(regression_results['selected_factors'])}")
    
    # Final formula and insights
    if formula_insights["status"] == "success":
        summary_parts.append(f"\n## FINAL FORMULA")
        summary_parts.append(formula_insights['simplified_formula'])
        
        summary_parts.append(f"\n## KEY INSIGHTS")
        for insight in formula_insights['insights']:
            summary_parts.append(f"• {insight}")
        
        if formula_insights['recommendations']:
            summary_parts.append(f"\n## RECOMMENDATIONS")
            for rec in formula_insights['recommendations']:
                summary_parts.append(f"• {rec}")
    
    return "\n".join(summary_parts) 