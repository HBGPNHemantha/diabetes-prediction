
# ==========================================
# 1. IMPORT DATASET
# ==========================================

diabetes <- read.csv("data/diabetes.csv", header = FALSE)

names(diabetes) <- c(
  "Pregnancies",
  "Glucose",
  "BloodPressure",
  "SkinThickness",
  "Insulin",
  "BMI",
  "Diabetes_Pedigree_Function",
  "Age",
  "Outcome"
)

# View dataset
head(diabetes)
dim(diabetes)
str(diabetes)
summary(diabetes)


# ==========================================
# 2. DATA CLEANING
# ==========================================

# Check missing values
colSums(is.na(diabetes))

# Check invalid zeros
columns <- c(
  "Glucose",
  "BloodPressure",
  "SkinThickness",
  "Insulin",
  "BMI"
)

colSums(diabetes[, columns] == 0)

# Replace invalid zeros with NA
for (col in columns) {
  diabetes[diabetes[[col]] == 0, col] <- NA
}

# Replace NA with median
for (col in columns) {
  median_value <- median(diabetes[[col]], na.rm = TRUE)
  diabetes[[col]][is.na(diabetes[[col]])] <- median_value
}

# Verify cleaning
colSums(is.na(diabetes))
colSums(diabetes == 0)


# ==========================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================

# Dataset information
dim(diabetes)
str(diabetes)
summary(diabetes)

# Outcome distribution
table(diabetes$Outcome)
prop.table(table(diabetes$Outcome)) * 100

# Select numerical features
numeric_cols <- setdiff(
  names(diabetes)[sapply(diabetes, is.numeric)],
  "Outcome"
)


# ==========================================
# 4. OUTLIER DETECTION
# ==========================================

for (col in numeric_cols) {
  
  Q1 <- quantile(diabetes[[col]], 0.25)
  Q3 <- quantile(diabetes[[col]], 0.75)
  
  IQR_value <- Q3 - Q1
  
  lower <- Q1 - 1.5 * IQR_value
  upper <- Q3 + 1.5 * IQR_value
  
  outliers <- sum(
    diabetes[[col]] < lower |
      diabetes[[col]] > upper
  )
  
  cat(col, ":", outliers, "outliers\n")
}


# ==========================================
# 5. DATA VISUALIZATION
# ==========================================

# Boxplots
par(mfrow = c(2, 4))

for (col in numeric_cols) {
  boxplot(
    diabetes[[col]],
    main = col,
    col = "lightblue",
    ylab = "Values"
  )
}

par(mfrow = c(1, 1))


# Histograms
par(mfrow = c(2, 4))

for (col in numeric_cols) {
  hist(
    diabetes[[col]],
    main = col,
    col = "lightblue",
    xlab = col,
    breaks = 20
  )
}

par(mfrow = c(1, 1))


# ==========================================
# 6. CORRELATION ANALYSIS
# ==========================================

# Calculate correlation matrix
cor_matrix <- cor(diabetes)

# Display correlation values
round(cor_matrix, 2)

# Correlation heatmap
library(corrplot)

corrplot(
  cor_matrix,
  method = "color",
  type = "upper",
  addCoef.col = "black",
  tl.col = "black",
  tl.srt = 45,
  tl.cex = 0.7,
  number.cex = 0.6,
  mar = c(0, 0, 2, 0)
)

# Correlation with Outcome
outcome_corr <- cor_matrix[
  "Outcome",
  setdiff(colnames(cor_matrix), "Outcome")
]

# Sort correlations
sort(outcome_corr, decreasing = TRUE)
