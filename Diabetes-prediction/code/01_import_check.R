
#Import dataset
diabetes <- read.csv("data/diabetes.csv",header = FALSE)


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

#Display first 6 records
head(diabetes)

#Display dataset dimensions
dim(diabetes)

#Display column names
names(diabetes)

#Check data types
str(diabetes)

#Summary
summary(diabetes)

#Check missing val
colSums(is.na(diabetes)) #is.na-identify missing val,colSum-take column sum

colSums(diabetes[, c(
  "Glucose",
  "BloodPressure",
  "SkinThickness",
  "Insulin",
  "BMI"
)] == 0)

# Replace invalid zero values with NA
columns <- c(
  "Glucose",
  "BloodPressure",
  "SkinThickness",
  "Insulin",
  "BMI"
)

for (col in columns) {
  diabetes[diabetes[[col]] == 0, col] <- NA
}

# Check missing values again
colSums(is.na(diabetes))



# Replace missing values using median

for (col in columns) {
  median_value <- median(diabetes[[col]], na.rm = TRUE)
  
  diabetes[[col]][is.na(diabetes[[col]])] <- median_value
}

# Check missing values again
colSums(is.na(diabetes))


# Dataset dimensions
dim(diabetes)

# Dataset structure
str(diabetes)

# Statistical summary
summary(diabetes)

# Count each outcome
table(diabetes$Outcome)

# Calculate percentages
prop.table(table(diabetes$Outcome)) * 100

# Check missing values in each column
colSums(is.na(diabetes))

# Count zero values in each column
colSums(diabetes == 0)