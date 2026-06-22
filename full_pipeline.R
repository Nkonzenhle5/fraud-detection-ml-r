# ==========================================
# FRAUD DETECTION PROJECT - CLEAN PIPELINE
# ==========================================

# ==========================================
# 1. LOAD LIBRARIES
# ==========================================
library(data.table)
library(dplyr)
library(ggplot2)
library(caret)
library(xgboost)

# ==========================================
# 2. LOAD DATA
# ==========================================
# Since your file is in the SAME folder, use this:
df <- fread("PS_20174392719_1491204439457_log.csv")

# Reduce dataset size for performance
df <- df[1:150000]

# ==========================================
# 3. DATA PREPARATION
# ==========================================

# Convert variables
df$isFraud <- as.factor(df$isFraud)
df$type <- as.factor(df$type)

# Remove unnecessary columns
df <- df %>% select(-nameOrig, -nameDest)

# ==========================================
# 4. FEATURE ENGINEERING
# ==========================================

# Convert to numeric for model
df$type <- as.numeric(df$type)
df$isFraud <- as.numeric(df$isFraud) - 1

# Create useful features
df$error_orig <- df$oldbalanceOrg - df$newbalanceOrig - df$amount
df$error_dest <- df$newbalanceDest - df$oldbalanceDest - df$amount

# Large transaction flag
df$is_large <- ifelse(df$amount > 200000, 1, 0)

# ==========================================

# 5. VISUALIZATIONS
# ==========================================

# Fraud distribution
ggplot(df, aes(x = as.factor(isFraud))) +
  geom_bar(fill = "steelblue") +
  ggtitle("Fraud vs Non-Fraud")

# Transaction amount distribution
ggplot(df, aes(x = amount)) +
  geom_histogram(bins = 50, fill = "orange") +
  ggtitle("Transaction Amount Distribution")

# ==========================================
# 6. TRAIN-TEST SPLIT
# ==========================================
set.seed(123)

train_index <- sample(1:nrow(df), 0.7 * nrow(df))

train <- df[train_index, ]
test  <- df[-train_index, ]

# Split features and target
X_train <- as.matrix(train %>% select(-isFraud))
y_train <- train$isFraud

X_test <- as.matrix(test %>% select(-isFraud))
y_test <- test$isFraud

# ==========================================
# 7. HANDLE IMBALANCE
# ==========================================
scale_pos_weight <- sum(y_train == 0) / sum(y_train == 1)

# ==========================================
# 8. MODEL (XGBOOST)
# ==========================================
model <- xgboost(
  data = X_train,
  label = y_train,
  nrounds = 100,
  objective = "binary:logistic",
  scale_pos_weight = scale_pos_weight,
  eval_metric = "auc",
  verbose = 0
)

# ==========================================
# 9. PREDICTIONS
# ==========================================
pred_prob <- predict(model, X_test)

# Adjust threshold for fraud detection
pred_class <- ifelse(pred_prob > 0.2, 1, 0)

# ==========================================
# 10. EVALUATION
# ==========================================
confusionMatrix(as.factor(pred_class), as.factor(y_test))

# ==========================================
# 11. FEATURE IMPORTANCE
# ==========================================
importance_matrix <- xgb.importance(model = model)

print(importance_matrix)

xgb.plot.importance(importance_matrix)
