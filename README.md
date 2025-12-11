# QST843-Amazon-Customer-Review-Analysis
Amazon Customer Review Analysis from BU QST 843 course - Hyun Sung (Evan) Park

## Personal Contributions

The following are the main tasks I contributed to this project:

### 1. Exploratory Data Analysis (EDA)
**File**: `Personal Contributions/QST5-EDA-evan.ipynb`

This notebook loads cleaned data, joins reviews with metadata, and performs the following exploratory data analyses:

#### EDA 1: Correlation Analysis between Verified Purchase and Rating
- **Objective**: Understand the impact of Verified Purchase on review ratings
- **Method**: 
  - Convert Verified Purchase (boolean) to numeric (0/1) by category
  - Calculate correlation coefficients by category using `F.corr("verified_purchase", "rating")`
  - Visualize with bar charts to analyze differences across categories
- **Results**: 
  - Overall very weak correlation (maximum 0.049)
  - Baby Products shows the highest positive correlation (0.049)
  - All Beauty shows negative correlation (-0.025)

#### EDA 2: Verified Purchase Skewness Analysis
- **Objective**: Analyze the difference in rating distributions between Verified Purchase and Non-Verified Purchase reviews
- **Method**:
  - Implement custom function `skew_by_category()`: Calculate skewness using the third moment of ratings
  - Filter Verified Purchase and Non-Verified Purchase reviews separately and calculate skewness for each
  - Compare skewness, average ratings, and review counts between the two groups by category
  - Visualize with comparative bar charts
- **Results**:
  - Musical Instruments, Baby Products, Office Products, Automotive, and Grocery categories show more pronounced skewness toward high ratings in Verified Purchase reviews
  - All Beauty shows relatively lower skewness for Verified Purchase reviews
  - Business Insight: Verified Purchase strategy may be more effective in certain categories

#### EDA 3: Helpful Votes and Rating Relationship Analysis
- **Objective**: Analyze how Helpful Votes concentrated on negative reviews (1-star) affect overall product ratings
- **Method**:
  - Calculate sum of Helpful Votes for 1-star and 5-star reviews by product (`parent_asin`)
  - Calculate `helpful_diff = helpful_sum_r5 - helpful_sum_r1`
  - Classify into 4 groups based on `helpful_diff`: "Mostly 1", "Balanced", "Slightly more on 5", "Mostly 5"
  - Calculate average rating for each group and visualize with bar charts
- **Results**:
  - Products with Helpful Votes concentrated on 1-star reviews have lower average ratings
  - The "Mostly 1" group has an average rating of approximately 2.72, which is very low
  - Business Insight: Helpful Votes concentration can be used as an early warning indicator

### 2. Machine Learning Analysis (ML Analysis)
**File**: `Personal Contributions/QST5-ML-analysis.ipynb`

#### Model 1: Average Rating Prediction Model (Random Forest Regressor)
- **Objective**: Predict average ratings based on product metadata and identify which factors influence ratings
- **Feature Engineering**:
  - **Product Name Features**: Word count (`name_word_count`), uppercase ratio (`caps_ratio`), special character presence (`has_special_chars`)
  - **Product Description**: Word count of description array (`description_words`)
  - **Media Flags**: Product image (`product_image_int`), product video (`product_video_int`)
  - **Category**: Transform `main_category` using StringIndexer and OneHotEncoder
  - **Price**: Direct use of `price` field
- **Model Configuration**:
  - Random Forest Regressor (numTrees=150, maxDepth=10, seed=42)
  - Train/Test split: 70/30
  - Pipeline: StringIndexer → OneHotEncoder → VectorAssembler → RandomForestRegressor
- **Model Performance**: 
  - RMSE: 0.675
  - R²: 0.031 (Low performance, but Feature Importance provides meaningful insights)
- **Feature Importance Analysis Results**:
  - **Product Video** has the greatest impact (importance 0.166) - Recommend investing in product videos
  - **Description Word Count** is second most important (importance 0.099) - Recommend writing detailed descriptions
  - **Price** has moderate impact (importance 0.085)
  - **Category**: "Cell Phones & Accessories" is the most influential category (importance 0.220)
  - Product name-related features have relatively low importance
- **Business Insights**:
  - Product video investment provides the highest ROI
  - Writing detailed product descriptions is important
  - Product name length or format has little impact on ratings

#### Model 2: Negative Review LDA Topic Modeling
- **Objective**: Analyze 1-2 star negative review texts to systematically classify the main causes of customer complaints
- **Data Preparation**:
  - Filter only 1-2 star reviews (`rating.isin([1, 2])`)
  - Use only reviews with non-null text
- **Text Preprocessing**:
  - Convert to lowercase
  - Remove HTML tags (`<[^>]+>`)
  - Remove URLs (`http\\S+`)
  - Expand English contractions (won't → will not, can't → can not, etc.)
  - Remove numbers and punctuation (`[^a-z\\s]`)
  - Remove 1-2 character short words (`\\b[a-z]{1,2}\\b`)
  - Normalize whitespace
- **Stopword Processing**:
  - Default English stopwords (StopWordsRemover)
  - Add custom stopwords (common words like product, item, amazon, seller)
  - Add Spanish stopwords (dataset contains Spanish reviews)
- **LDA Modeling**:
  - Tokenizer → StopWordsRemover → CountVectorizer (vocabSize=10000, minDF=10) → LDA
  - Number of topics: 30
  - maxIter: 50
- **Topic Classification and Interpretation**:
  - Classify 30 topics into 6 groups:
    - **Group A: Tech & Functionality** - Charging/battery, printer, screen protection, audio/buttons, SIM card, etc.
    - **Group B: Durability & Quality** - Early breakage, poor quality materials, short lifespan, etc.
    - **Group C: Assembly & Installation** - Assembly issues, adhesion failure, missing manuals/parts, etc.
    - **Group D: Shipping & Misleading** - Shipping damage, description/color mismatch, return/refund issues, etc.
    - **Group E: Domain-Specific Complaints** - Food/taste, pet supplies, stationery, gardening products, etc.
    - **Group F: Fitment & Sizing** - Phone cases, vehicle accessories, cover sizes, etc.
  - Provide keywords and business insights for each topic
  - Visualization: Generate topic category keyword table (`lda_topics_table.png`)
- **Business Insights**:
  - Sellers can perform the same analysis on their own product groups to identify specific complaint patterns
  - Enable systematic product improvement plans through root cause analysis

### Contributions to Final Submission File

All of the above work has been integrated into the final submission file `QST5-Amazon-Customer-Review-Analysis.ipynb`:

- **EDA Section**: 
  - "4. Does helpful votes on negative reviews indicate low ratings?" - Includes Helpful Votes analysis
  - Verified Purchase related analysis is included as insights in the conclusion section of the final report
  
- **ML Section**: 
  - "Model 1) Average rating prediction" - Includes Random Forest Regressor model and Feature Importance analysis
  - "Model 2) Negative Review NLP Analysis" - Includes LDA topic modeling
  - Includes Feature Importance visualizations and business recommendations

### Key Achievements and Contributions

1. **Application of Statistical Analysis Methodologies**: 
   - Discovered category-specific differences through correlation analysis between Verified Purchase and Rating
   - Quantified asymmetry in rating distributions through skewness calculation
   - Proposed methodology for using Helpful Votes concentration as an early warning indicator

2. **Machine Learning Model Development and Interpretation**:
   - Derived actionable insights through Feature Importance analysis using Random Forest model
   - Discovered that product videos have the greatest impact on ratings
   - Provided practical recommendations through Feature Importance despite low model performance

3. **NLP-based Customer Complaint Analysis**:
   - Extracted 30 topics through LDA topic modeling of 1-2 star negative reviews
   - Systematic classification into 6 groups enables root cause analysis
   - Presented practical analysis methodology applicable to sellers' own products

4. **Actionable Business Insights**:
   - Recommended category-specific differentiation of Verified Purchase strategies
   - Proposed building early warning systems for product quality through Helpful Votes monitoring
   - Product content optimization strategies (video investment, detailed description writing)
   - Product improvement prioritization through negative review pattern analysis
