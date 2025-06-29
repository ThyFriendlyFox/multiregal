"""
Comprehensive data cleaning module for MultiRegal.
Handles various types of dirty data including missing values, outliers, 
inconsistent formatting, mixed data types, and more.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
import warnings


class DataCleaner:
    """Comprehensive data cleaning class for regression analysis."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.cleaning_log = []
        
    def log(self, message: str):
        """Add message to cleaning log."""
        if self.verbose:
            print(f"🧹 {message}")
        self.cleaning_log.append(message)
    
    def clean_dataset(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Comprehensive data cleaning pipeline.
        
        Args:
            df: Raw DataFrame
            target_column: Name of target variable
            
        Returns:
            Tuple of (cleaned_df, cleaning_report)
        """
        if df.empty:
            return df, {"status": "error", "message": "Empty dataset provided"}
            
        original_shape = df.shape
        self.log(f"Starting with dataset: {original_shape[0]} rows, {original_shape[1]} columns")
        
        cleaning_report = {
            "original_shape": original_shape,
            "steps_performed": [],
            "issues_found": [],
            "columns_modified": [],
            "rows_removed": 0,
            "cleaning_log": []
        }
        
        try:
            # Step 1: Handle duplicate rows
            df_clean, duplicates_info = self._remove_duplicates(df)
            if duplicates_info["duplicates_removed"] > 0:
                cleaning_report["steps_performed"].append("duplicate_removal")
                cleaning_report["issues_found"].append(f"Found {duplicates_info['duplicates_removed']} duplicate rows")
            
            # Step 2: Clean column names first
            df_clean, column_cleaning = self._clean_column_names(df_clean)
            if column_cleaning["columns_renamed"]:
                cleaning_report["steps_performed"].append("column_name_cleaning")
                cleaning_report["columns_modified"].extend(column_cleaning["columns_renamed"])
            
            # Step 3: Map target column to cleaned name
            original_target = target_column
            self.log(f"Looking for target column '{original_target}' in original columns: {list(df.columns)}")
            
            # Find the cleaned version of the target column
            target_column_clean = None
            column_mapping = list(zip(df.columns, df_clean.columns))
            self.log(f"Column mapping: {column_mapping}")
            
            for original, cleaned in column_mapping:
                if original == target_column:
                    target_column_clean = cleaned
                    self.log(f"Found mapping: '{original}' -> '{cleaned}'")
                    break
            
            if target_column_clean is None:
                # Try case-insensitive matching as fallback
                for original, cleaned in column_mapping:
                    if original.lower() == target_column.lower():
                        target_column_clean = cleaned
                        self.log(f"Found case-insensitive mapping: '{original}' -> '{cleaned}'")
                        break
            
            if target_column_clean is None:
                return df, {
                    "status": "error", 
                    "message": f"Target column '{original_target}' not found after cleaning. Available columns: {list(df_clean.columns)}"
                }
            
            # Update target column name to cleaned version
            target_column = target_column_clean
            self.log(f"Target column '{original_target}' mapped to '{target_column_clean}'")
            
            # Step 4: Handle mixed data types and convert to numeric
            df_clean, conversion_info = self._handle_mixed_types(df_clean, target_column)
            if conversion_info["columns_converted"]:
                cleaning_report["steps_performed"].append("data_type_conversion")
                cleaning_report["columns_modified"].extend(conversion_info["columns_converted"])
                cleaning_report["issues_found"].extend(conversion_info["issues_found"])
            
            # Step 5: Handle missing values
            df_clean, missing_info = self._handle_missing_values(df_clean, target_column)
            if missing_info["columns_imputed"]:
                cleaning_report["steps_performed"].append("missing_value_imputation")
                cleaning_report["columns_modified"].extend(missing_info["columns_imputed"])
                cleaning_report["issues_found"].extend(missing_info["issues_found"])
            
            # Step 6: Remove outliers
            df_clean, outlier_info = self._handle_outliers(df_clean, target_column)
            if outlier_info["outliers_removed"] > 0:
                cleaning_report["steps_performed"].append("outlier_removal")
                cleaning_report["rows_removed"] += outlier_info["outliers_removed"]
                cleaning_report["issues_found"].append(f"Removed {outlier_info['outliers_removed']} outlier rows")
            
            # Step 7: Final validation
            df_clean, validation_info = self._final_validation(df_clean, target_column)
            if validation_info["rows_removed"] > 0:
                cleaning_report["steps_performed"].append("final_validation")
                cleaning_report["rows_removed"] += validation_info["rows_removed"]
            
            # Update cleaning report
            cleaning_report["final_shape"] = df_clean.shape
            cleaning_report["cleaning_log"] = self.cleaning_log
            cleaning_report["status"] = "success"
            
            self.log(f"Cleaning complete: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
            
            return df_clean, cleaning_report
            
        except Exception as e:
            return df, {
                "status": "error",
                "message": f"Data cleaning failed: {str(e)}",
                "cleaning_log": self.cleaning_log
            }
    
    def _remove_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Remove duplicate rows."""
        original_len = len(df)
        df_clean = df.drop_duplicates()
        duplicates_removed = original_len - len(df_clean)
        
        if duplicates_removed > 0:
            self.log(f"Removed {duplicates_removed} duplicate rows")
            
        return df_clean, {"duplicates_removed": duplicates_removed}
    
    def _clean_column_names(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Clean and standardize column names."""
        original_columns = df.columns.tolist()
        df_clean = df.copy()
        
        # Clean column names
        new_columns = []
        columns_renamed = []
        
        for col in df.columns:
            # Remove special characters, convert to lowercase, replace spaces with underscores
            clean_col = re.sub(r'[^\w\s]', '', str(col))
            clean_col = re.sub(r'\s+', '_', clean_col.strip())
            clean_col = clean_col.lower()
            
            # Handle empty column names
            if not clean_col:
                clean_col = f"column_{len(new_columns)}"
            
            # Handle duplicate column names
            base_col = clean_col
            counter = 1
            while clean_col in new_columns:
                clean_col = f"{base_col}_{counter}"
                counter += 1
            
            new_columns.append(clean_col)
            if clean_col != col:
                columns_renamed.append(f"{col} -> {clean_col}")
        
        df_clean.columns = new_columns
        
        if columns_renamed:
            self.log(f"Renamed {len(columns_renamed)} columns for consistency")
            
        return df_clean, {"columns_renamed": columns_renamed}
    
    def _handle_mixed_types(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, Dict]:
        """Handle mixed data types and convert to numeric where possible."""
        df_clean = df.copy()
        conversion_info = {"columns_converted": [], "issues_found": []}
        
        for col in df_clean.columns:
            if col == target_column:
                continue
                
            try:
                # Check if column is already numeric
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    continue
                
                # Try to convert to numeric
                original_col = df_clean[col].copy()
                
                # Handle common string representations of numbers
                if df_clean[col].dtype == 'object':
                    # Remove common prefixes/suffixes and formatting
                    df_clean[col] = df_clean[col].astype(str)
                    df_clean[col] = df_clean[col].str.replace(r'[$,€£¥]', '', regex=True)  # Currency symbols
                    df_clean[col] = df_clean[col].str.replace(r'[%]', '', regex=True)  # Percentage
                    df_clean[col] = df_clean[col].str.replace(r'[,]', '', regex=True)  # Thousands separator
                    df_clean[col] = df_clean[col].str.strip()  # Whitespace
                    
                    # Handle boolean-like strings
                    bool_map = {'true': 1, 'false': 0, 'yes': 1, 'no': 0, 'y': 1, 'n': 0}
                    lower_col = df_clean[col].str.lower()
                    if lower_col.isin(bool_map.keys()).any():
                        df_clean[col] = lower_col.map(bool_map).fillna(df_clean[col])
                
                # Attempt numeric conversion
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                
                # Check conversion success
                non_null_before = original_col.notna().sum()
                non_null_after = df_clean[col].notna().sum()
                
                if non_null_after < non_null_before * 0.5:  # Lost more than 50% of data
                    # Revert conversion
                    df_clean[col] = original_col
                    conversion_info["issues_found"].append(
                        f"Column '{col}': Could not convert to numeric (would lose {non_null_before - non_null_after} values)"
                    )
                else:
                    conversion_info["columns_converted"].append(col)
                    if non_null_after < non_null_before:
                        lost_values = non_null_before - non_null_after
                        self.log(f"Converted '{col}' to numeric (lost {lost_values} non-numeric values)")
                    else:
                        self.log(f"Converted '{col}' to numeric")
                        
            except Exception as e:
                conversion_info["issues_found"].append(f"Column '{col}': Conversion failed - {str(e)}")
        
        return df_clean, conversion_info
    
    def _handle_missing_values(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, Dict]:
        """Handle missing values with intelligent imputation."""
        df_clean = df.copy()
        missing_info = {"columns_imputed": [], "issues_found": []}
        
        # Check overall missing value percentage
        total_cells = df_clean.shape[0] * df_clean.shape[1]
        missing_cells = df_clean.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100
        
        if missing_percentage > 0:
            self.log(f"Dataset has {missing_percentage:.1f}% missing values")
        
        # Handle target column missing values
        target_missing = df_clean[target_column].isnull().sum()
        if target_missing > 0:
            # Remove rows with missing target values
            df_clean = df_clean.dropna(subset=[target_column])
            self.log(f"Removed {target_missing} rows with missing target values")
            missing_info["issues_found"].append(f"Removed {target_missing} rows with missing target")
        
        # Handle feature columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        for col in numeric_cols:
            missing_count = df_clean[col].isnull().sum()
            missing_pct = (missing_count / len(df_clean)) * 100
            
            if missing_count == 0:
                continue
                
            if missing_pct > 50:
                # Drop columns with >50% missing values
                df_clean = df_clean.drop(columns=[col])
                self.log(f"Dropped column '{col}' (missing {missing_pct:.1f}% of values)")
                missing_info["issues_found"].append(f"Dropped column '{col}' due to excessive missing values")
                continue
            
            # Impute missing values
            if missing_pct > 0:
                # Use median for skewed data, mean for normal data
                if abs(df_clean[col].skew()) > 1:
                    fill_value = df_clean[col].median()
                    method = "median"
                else:
                    fill_value = df_clean[col].mean()
                    method = "mean"
                
                df_clean[col] = df_clean[col].fillna(fill_value)
                self.log(f"Imputed {missing_count} missing values in '{col}' using {method}")
                missing_info["columns_imputed"].append(col)
        
        return df_clean, missing_info
    
    def _handle_outliers(self, df: pd.DataFrame, target_column: str, 
                        method: str = "iqr", threshold: float = 3.0) -> Tuple[pd.DataFrame, Dict]:
        """Remove outliers using IQR or Z-score method."""
        df_clean = df.copy()
        outlier_info = {"outliers_removed": 0, "columns_processed": []}
        
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create a mask for outliers
        outlier_mask = pd.Series([False] * len(df_clean), index=df_clean.index)
        
        for col in numeric_cols:
            if df_clean[col].nunique() < 2:  # Skip constant columns
                continue
                
            if method == "iqr":
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                col_outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                
            else:  # z-score method
                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                col_outliers = pd.Series([False] * len(df_clean), index=df_clean.index)
                valid_idx = df_clean[col].dropna().index
                col_outliers.loc[valid_idx] = z_scores > threshold
            
            outlier_mask |= col_outliers
            
            if col_outliers.sum() > 0:
                outlier_info["columns_processed"].append(col)
        
        # Remove outlier rows, but keep at least 10 rows minimum
        outliers_to_remove = outlier_mask.sum()
        remaining_rows = len(df_clean) - outliers_to_remove
        
        if remaining_rows >= 10 and outliers_to_remove > 0:
            df_clean = df_clean[~outlier_mask]
            outlier_info["outliers_removed"] = outliers_to_remove
            self.log(f"Removed {outliers_to_remove} outlier rows using {method} method")
        elif outliers_to_remove > 0:
            self.log(f"Skipped outlier removal (would leave only {remaining_rows} rows)")
        
        return df_clean, outlier_info
    
    def _final_validation(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, Dict]:
        """Final validation and cleanup."""
        df_clean = df.copy()
        validation_info = {"rows_removed": 0, "issues_found": []}
        
        original_len = len(df_clean)
        
        # Remove rows where all feature values are the same (no variation)
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        if len(numeric_cols) > 1:
            # Check for rows with no variation in features
            feature_data = df_clean[numeric_cols]
            constant_rows = feature_data.nunique(axis=1) == 1
            
            if constant_rows.any():
                df_clean = df_clean[~constant_rows]
                removed = constant_rows.sum()
                validation_info["rows_removed"] = removed
                self.log(f"Removed {removed} rows with constant feature values")
        
        # Ensure minimum dataset size
        if len(df_clean) < 10:
            validation_info["issues_found"].append("Dataset too small after cleaning (less than 10 rows)")
        
        # Ensure we have at least 2 numeric columns for regression
        final_numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if len(final_numeric_cols) < 2:
            validation_info["issues_found"].append("Not enough numeric columns for regression analysis")
        
        return df_clean, validation_info
    
    def generate_cleaning_report(self, cleaning_results: Dict[str, Any]) -> str:
        """Generate a human-readable cleaning report."""
        if cleaning_results["status"] != "success":
            return f"❌ Data cleaning failed: {cleaning_results.get('message', 'Unknown error')}"
        
        report = ["📋 DATA CLEANING REPORT", "=" * 30]
        
        # Overview
        original_shape = cleaning_results["original_shape"]
        final_shape = cleaning_results["final_shape"]
        
        report.append(f"\n📊 OVERVIEW:")
        report.append(f"  Original: {original_shape[0]} rows × {original_shape[1]} columns")
        report.append(f"  Final: {final_shape[0]} rows × {final_shape[1]} columns")
        report.append(f"  Rows removed: {cleaning_results['rows_removed']}")
        
        # Steps performed
        if cleaning_results["steps_performed"]:
            report.append(f"\n🔧 CLEANING STEPS PERFORMED:")
            step_names = {
                "duplicate_removal": "Removed duplicate rows",
                "column_name_cleaning": "Standardized column names",
                "data_type_conversion": "Converted data types to numeric",
                "missing_value_imputation": "Imputed missing values",
                "outlier_removal": "Removed statistical outliers",
                "final_validation": "Final data validation"
            }
            for step in cleaning_results["steps_performed"]:
                report.append(f"  ✅ {step_names.get(step, step)}")
        
        # Issues found
        if cleaning_results["issues_found"]:
            report.append(f"\n⚠️  ISSUES ADDRESSED:")
            for issue in cleaning_results["issues_found"]:
                report.append(f"  • {issue}")
        
        # Columns modified
        if cleaning_results["columns_modified"]:
            report.append(f"\n📝 COLUMNS MODIFIED:")
            unique_cols = list(set(cleaning_results["columns_modified"]))
            for col in unique_cols[:10]:  # Show first 10
                report.append(f"  • {col}")
            if len(unique_cols) > 10:
                report.append(f"  ... and {len(unique_cols) - 10} more")
        
        return "\n".join(report)


def clean_data_for_analysis(data_input: str, target_column: str, 
                          verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main function to clean data for regression analysis.
    
    Args:
        data_input: CSV file path or CSV content string
        target_column: Name of target variable
        verbose: Whether to print cleaning progress
        
    Returns:
        Tuple of (cleaned_dataframe, cleaning_report)
    """
    try:
        # Load the data
        try:
            if data_input.endswith('.csv'):
                df = pd.read_csv(data_input)
            else:
                from io import StringIO
                df = pd.read_csv(StringIO(data_input))
        except:
            from io import StringIO
            df = pd.read_csv(StringIO(data_input))
        
        # Initialize cleaner and perform cleaning
        cleaner = DataCleaner(verbose=verbose)
        cleaned_df, cleaning_report = cleaner.clean_dataset(df, target_column)
        
        return cleaned_df, cleaning_report
        
    except Exception as e:
        return pd.DataFrame(), {
            "status": "error",
            "message": f"Failed to load or clean data: {str(e)}"
        } 