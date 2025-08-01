# Gurgaon Real Estate Data Analysis

A comprehensive data analysis project focused on exploring, cleaning, and analyzing real estate data from Gurgaon. This project demonstrates various data preprocessing techniques including exploratory data analysis, missing value handling, and outlier detection and treatment.

# Project Overview

This project performs an in-depth analysis of the Gurgaon real estate dataset, implementing various data science techniques to understand property trends, price distributions, and market characteristics. The analysis covers data exploration, cleaning, and preprocessing steps essential for any machine learning pipeline.

# Objectives

- **Data Exploration**: Comprehensive analysis of property types, prices, areas, and other features
- **Data Cleaning**: Handle missing values and duplicate records
- **Outlier Detection**: Identify anomalies using statistical methods (Z-score, IQR)
- **Outlier Treatment**: Apply various strategies including winsorization, trimming, and transformation
- **Visualization**: Create insightful plots to understand data distributions and relationships

# Technologies Used

- **Python 3.x**
- **Libraries**:
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computations
  - `matplotlib` - Data visualization
  - `seaborn` - Statistical data visualization
  - `scikit-learn` - Machine learning preprocessing tools
  - `scipy` - Statistical functions and tests

# Project Structure

```
G_003_016_121_150/
├── CODE/
│   ├── G_003_016_121_150_code.ipynb.ipynb
│   └── ML PROJECT.py.py
├── REPORT/
│   ├── G_003_016_121_150_Report.pdf
│   ├── G_003_016_121_150_Report_source_docx_latex.docx
│   └── README.txt
└── README.md
```

# Getting Started

### Prerequisites

Make sure you have Python 3.x installed along with the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd G_003_016_121_150
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your dataset (`Gurgaon_RealEstate.csv`) in the appropriate directory or update the file path in the code.

### Usage

Run the main Python script:
```bash
python "ML PROJECT.py.py"
```

Or open and run the Jupyter notebook:
```bash
jupyter notebook "G_003_016_121_150_code.ipynb.ipynb"
```

# Analysis Tasks

### Task 1: Exploratory Data Analysis (EDA)
- **Data Loading**: Import and examine the Gurgaon real estate dataset
- **Data Overview**: Check data types, shapes, and basic information
- **Duplicate Detection**: Identify and remove duplicate records
- **Feature Exploration**:
  - Property type distribution (flats vs houses)
  - Society analysis with threshold filtering
  - Price distribution analysis
  - Area, bedroom, and bathroom distributions
  - Statistical measures (skewness, kurtosis)
- **Multivariate Analysis**: 
  - Property type vs price relationships
  - Price vs area scatter plots

### Task 2: Missing Value Handling
- **Missing Value Detection**: Identify null values across all columns
- **Imputation Strategies**:
  - Mean imputation for numerical columns
  - Mode imputation for categorical columns
- **Visualization**: Heatmaps to visualize missing data patterns

### Task 3: Outlier Detection
- **Statistical Methods**:
  - **Z-score method**: Identify values beyond 3 standard deviations
  - **IQR method**: Detect outliers using interquartile range
- **Visualization**: Histograms and box plots for outlier identification

### Task 4: Outlier Treatment
- **Treatment Strategies**:
  - **Winsorization**: Cap extreme values at 5th and 95th percentiles
  - **Trimming**: Remove outlier observations
  - **Log Transformation**: Apply logarithmic transformation
- **Impact Assessment**: Compare metrics before and after treatment

# Key Features Analyzed

- **property_type**: Type of property (flat/house)
- **society**: Housing society name
- **price**: Property price
- **price_per_sqft**: Price per square foot
- **area**: Total area of the property
- **bedRoom**: Number of bedrooms
- **bathroom**: Number of bathrooms
- **carpet_area**: Carpet area of the property
- **built_up_area**: Built-up area
- **super_built_up_area**: Super built-up area

# Results and Insights

The analysis provides:
- Comprehensive understanding of Gurgaon real estate market trends
- Clean, preprocessed dataset ready for machine learning models
- Statistical insights into property characteristics
- Effective outlier treatment strategies
- Visualization of data distributions and relationships

# Team Contributors

| Name | Student ID | Contribution |
|------|------------|--------------|
| A. CHARITHA REDDY | SE22UCSE016 | 25% |
| ABHIGNA KAPPAGANTULA | SE22UCSE003 | 25% |
| SRUTHI LINGAMPALLI | SE22UCSE150 | 25% |
| K. RITHWIKA | SE22UCSE121 | 25% |

# Documentation

- **Detailed Report**: `G_003_016_121_150_Report.pdf`
- **Source Document**: `G_003_016_121_150_Report_source_docx_latex.docx`

# Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

# License

This project is part of an academic assignment. Please respect academic integrity guidelines when using this code.

# Additional Notes

- Ensure the dataset path is correctly specified in the code
- The project demonstrates best practices in data preprocessing and analysis
- All visualizations are designed to provide clear insights into the data characteristics
- The code includes comprehensive comments for better understanding

---

**Note**: This project is designed for educational purposes and demonstrates various data science techniques applied to real estate data analysis.
