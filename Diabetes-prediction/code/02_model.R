
# ==========================================
# DIABETES PREDICTION - MACHINE LEARNING
# ==========================================
# Using: Logistic Regression + Random Forest
# Dataset: Pima Indians Diabetes Dataset
# ==========================================


# ---- Load required packages ----
library(caret)
library(randomForest)
library(ggplot2)
library(corrplot)
library(dplyr)


# ==========================================
# STEP 1: IMPORT DATASET
# ==========================================

diabetes <- read.csv("data/diabetes.csv", header = FALSE)

names(diabetes) <- c(
  "Pregnancies", "Glucose", "BloodPressure",
  "SkinThickness", "Insulin", "BMI",
  "DiabetesPedigreeFunction", "Age", "Outcome"
)

cat("Dataset Dimensions:", dim(diabetes), "\n")
head(diabetes)
str(diabetes)
summary(diabetes)


# ==========================================
# STEP 2: DATA CLEANING & PREPROCESSING
# ==========================================

# Check missing values
cat("\nMissing values before cleaning:\n")
print(colSums(is.na(diabetes)))

# Columns where 0 is medically invalid
invalid_cols <- c("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")

cat("\nZero counts in medical columns:\n")
print(colSums(diabetes[, invalid_cols] == 0))

# Replace invalid zeros with NA
for (col in invalid_cols) {
  diabetes[diabetes[[col]] == 0, col] <- NA
}

# Impute NA with column median
for (col in invalid_cols) {
  med <- median(diabetes[[col]], na.rm = TRUE)
  diabetes[[col]][is.na(diabetes[[col]])] <- med
}

# Verify: no NAs remain
cat("\nMissing values after imputation:\n")
print(colSums(is.na(diabetes)))

# Convert Outcome to factor
diabetes$Outcome <- factor(diabetes$Outcome, levels = c(0, 1),
                           labels = c("No", "Yes"))


# ==========================================
# STEP 3: EXPLORATORY DATA ANALYSIS
# ==========================================

cat("\nOutcome distribution:\n")
print(table(diabetes$Outcome))
print(prop.table(table(diabetes$Outcome)) * 100)

# Numeric columns (excluding Outcome)
num_cols <- names(diabetes)[sapply(diabetes, is.numeric)]

# Summary statistics by Outcome
cat("\nGroup means by Outcome:\n")
diabetes %>%
  group_by(Outcome) %>%
  summarise(across(all_of(num_cols), mean, .names = "mean_{.col}")) %>%
  print()


# ==========================================
# STEP 4: VISUALIZATIONS
# ==========================================

dir.create("figures", showWarnings = FALSE)

# --- 4a. Outcome bar chart ---
p1 <- ggplot(diabetes, aes(x = Outcome, fill = Outcome)) +
  geom_bar() +
  scale_fill_manual(values = c("steelblue", "tomato")) +
  labs(title = "Diabetes Outcome Distribution",
       x = "Diabetes", y = "Count") +
  theme_minimal()

ggsave("figures/01_outcome_distribution.png", p1, width = 6, height = 4)

# --- 4b. Glucose by Outcome (boxplot) ---
p2 <- ggplot(diabetes, aes(x = Outcome, y = Glucose, fill = Outcome)) +
  geom_boxplot() +
  scale_fill_manual(values = c("steelblue", "tomato")) +
  labs(title = "Glucose Levels by Diabetes Outcome",
       x = "Diabetes", y = "Glucose") +
  theme_minimal()

ggsave("figures/02_glucose_boxplot.png", p2, width = 6, height = 4)

# --- 4c. BMI by Outcome (boxplot) ---
p3 <- ggplot(diabetes, aes(x = Outcome, y = BMI, fill = Outcome)) +
  geom_boxplot() +
  scale_fill_manual(values = c("steelblue", "tomato")) +
  labs(title = "BMI by Diabetes Outcome",
       x = "Diabetes", y = "BMI") +
  theme_minimal()

ggsave("figures/03_bmi_boxplot.png", p3, width = 6, height = 4)

# --- 4d. Age distribution ---
p4 <- ggplot(diabetes, aes(x = Age, fill = Outcome)) +
  geom_histogram(bins = 20, position = "dodge", alpha = 0.8) +
  scale_fill_manual(values = c("steelblue", "tomato")) +
  labs(title = "Age Distribution by Diabetes Outcome",
       x = "Age", y = "Count") +
  theme_minimal()

ggsave("figures/04_age_distribution.png", p4, width = 6, height = 4)

# --- 4e. Correlation heatmap ---
png("figures/05_correlation_heatmap.png", width = 700, height = 600)
cor_mat <- cor(diabetes %>% mutate(Outcome = as.numeric(Outcome)))
corrplot(cor_mat, method = "color", type = "upper",
         addCoef.col = "black", tl.col = "black",
         tl.srt = 45, tl.cex = 0.8, number.cex = 0.65,
         mar = c(0, 0, 2, 0),
         title = "Feature Correlation Matrix")
dev.off()

cat("All figures saved to figures/\n")


# ==========================================
# STEP 5: TRAIN / TEST SPLIT
# ==========================================

set.seed(42)

train_idx <- createDataPartition(diabetes$Outcome, p = 0.8, list = FALSE)
train_data <- diabetes[train_idx, ]
test_data  <- diabetes[-train_idx, ]

cat("\nTraining set size:", nrow(train_data), "\n")
cat("Testing set size: ", nrow(test_data),  "\n")


# ==========================================
# STEP 6: MODEL TRAINING
# ==========================================

# Cross-validation setup (5-fold)
ctrl <- trainControl(method = "cv", number = 5,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary,
                     savePredictions = TRUE)

# --- Model A: Logistic Regression ---
cat("\nTraining Logistic Regression...\n")
set.seed(42)
model_lr <- train(Outcome ~ ., data = train_data,
                  method = "glm", family = "binomial",
                  trControl = ctrl, metric = "ROC")

cat("Logistic Regression CV ROC:", round(max(model_lr$results$ROC), 4), "\n")

# --- Model B: Random Forest ---
cat("\nTraining Random Forest...\n")
set.seed(42)
model_rf <- train(Outcome ~ ., data = train_data,
                  method = "rf",
                  trControl = ctrl, metric = "ROC",
                  tuneLength = 3)

cat("Random Forest CV ROC:", round(max(model_rf$results$ROC), 4), "\n")


# ==========================================
# STEP 7: MODEL EVALUATION
# ==========================================

evaluate_model <- function(model, test_data, model_name) {
  preds <- predict(model, test_data)
  cm    <- confusionMatrix(preds, test_data$Outcome, positive = "Yes")

  cat("\n========================================\n")
  cat("Model:", model_name, "\n")
  cat("========================================\n")
  print(cm)

  cat("\nAccuracy :", round(cm$overall["Accuracy"], 4), "\n")
  cat("Sensitivity (Recall):", round(cm$byClass["Sensitivity"], 4), "\n")
  cat("Specificity:         ", round(cm$byClass["Specificity"], 4), "\n")
  cat("Precision:           ", round(cm$byClass["Pos Pred Value"], 4), "\n")
  cat("F1 Score:            ", round(cm$byClass["F1"], 4), "\n")

  return(cm)
}

cm_lr <- evaluate_model(model_lr, test_data, "Logistic Regression")
cm_rf <- evaluate_model(model_rf, test_data, "Random Forest")


# ==========================================
# STEP 8: FEATURE IMPORTANCE (Random Forest)
# ==========================================

importance_df <- varImp(model_rf)$importance
importance_df$Feature <- rownames(importance_df)
colnames(importance_df)[1] <- "Importance"
importance_df <- importance_df[order(-importance_df$Importance), ]

cat("\nFeature Importance (Random Forest):\n")
print(importance_df)

p5 <- ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Feature Importance - Random Forest",
       x = "Feature", y = "Importance") +
  theme_minimal()

ggsave("figures/06_feature_importance.png", p5, width = 7, height = 5)


# ==========================================
# STEP 9: MODEL COMPARISON SUMMARY
# ==========================================

acc_lr <- confusionMatrix(predict(model_lr, test_data),
                          test_data$Outcome)$overall["Accuracy"]
acc_rf <- confusionMatrix(predict(model_rf, test_data),
                          test_data$Outcome)$overall["Accuracy"]

comparison <- data.frame(
  Model    = c("Logistic Regression", "Random Forest"),
  CV_ROC   = c(round(max(model_lr$results$ROC), 4),
               round(max(model_rf$results$ROC), 4)),
  Test_Acc = c(round(acc_lr, 4), round(acc_rf, 4))
)

cat("\n========================================\n")
cat("MODEL COMPARISON SUMMARY\n")
cat("========================================\n")
print(comparison)

best_model <- ifelse(acc_rf >= acc_lr, "Random Forest", "Logistic Regression")
cat("\nBest Model:", best_model, "\n")

cat("\nAnalysis complete. All figures saved to figures/\n")

