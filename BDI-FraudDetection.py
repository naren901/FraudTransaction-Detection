# Databricks notebook source
# MAGIC %md
# MAGIC ## Analyzing Card Trnsaction for identifying Fraud Transactions

# COMMAND ----------

# MAGIC %md
# MAGIC This data is made availble by "Vesta Corporation" for the "IEEE-CIS Fraud Detection" competetion held on Kaggle.<br>
# MAGIC Credit/Debit card transactions are analyzed and flagged as Fraud or Not Fraud, if the transaction is identified as Fraud transaction the card owner is alerted.<br>
# MAGIC In this Analysis we've tried to identify if a transaction is Fraud or Not Fraud.<br>
# MAGIC <br>
# MAGIC A decision tree model, Logistic regression model is used to identify a Fraud transaction.

# COMMAND ----------

# MAGIC %pip install mlflow

# COMMAND ----------

import shutil
import mlflow
import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.types as st
from pyspark.sql.types import StructType,StructField, StringType
from pyspark.sql.functions import isnan, when, count, col, round
from pyspark.ml import Pipeline, Model
from pyspark.ml.feature import StringIndexer, IndexToString, VectorAssembler, OneHotEncoder, OneHotEncoderModel
from distutils.version import LooseVersion

# COMMAND ----------

# MAGIC %md
# MAGIC ##DATA
# MAGIC Kaggle competetion link: https://www.kaggle.com/c/ieee-fraud-detection/data <br>
# MAGIC Training and Testing datasets are provided by Vesta Corp, the datasets are split into Transaction and Identity datasets.<br>
# MAGIC Training dataset is labelled and Testing dataset is not labelled. For the analysis Training dataset is used to train and test the models in this analysis.<br>
# MAGIC Each of the Training and Testing dataset are split into Identity and Transaction dataset are joined by matching the TransactionID.<br>
# MAGIC Phone verification is required for downloading the data from Kaggle website, for this analysis downloaded datasets are uploaded to googledrive and a link to download those files.<br>

# COMMAND ----------

# MAGIC %md
# MAGIC #####<b> TRANSACTION  DATASET FEATURES </b> <br>
# MAGIC TransactionDT -  Timedelta from a given reference datetime (not an actual timestamp) <br>
# MAGIC TransactionAMT -  transaction payment amount in USD <br>
# MAGIC ProductCD  - product code, the product for each transaction  <br>
# MAGIC card1 - card6 payment card information, such as card type, card category, issue bank, country, etc.  <br>
# MAGIC addr - address, dist - distance  <br>
# MAGIC P_ and (R__) email domain - purchaser and recipient email domain  <br>
# MAGIC C1-C14 - counting, such as how many addresses are found to be associated with the payment card, etc. The actual meaning is masked.  <br>
# MAGIC D1-D15 - timedelta, such as days between previous transaction, etc.  <br>
# MAGIC M1-M9 - match, such as names on card and address, etc.  <br>
# MAGIC Vxxx - Vesta engineered features, including ranking, counting, other entity relations.  <br>

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Download the Training data

# COMMAND ----------

# MAGIC %sh
# MAGIC wget https://www.dropbox.com/s/zmfx8hrt14dlh6y/train_identity.csv?dl=0
# MAGIC wget https://www.dropbox.com/s/0cmnfg35uekw9cz/train_transaction.csv?dl=0
# MAGIC ls -all

# COMMAND ----------

# MAGIC %md
# MAGIC #####Rename the downloaded files

# COMMAND ----------

# MAGIC %sh
# MAGIC mv train_identity.csv?dl=0 train_identity.csv
# MAGIC mv train_transaction.csv?dl=0 train_transaction.csv
# MAGIC ls -all

# COMMAND ----------

transaction_file_path = 'file:/databricks/driver/train_transaction.csv'   #'dbfs:/FileStore/tables/train_transaction.csv'
identity_file_path = 'file:/databricks/driver/train_identity.csv' #'dbfs:/FileStore/tables/train_identity.csv'

# COMMAND ----------

# MAGIC %md ###Load Transaction and Identity data into Spark DataFrames

# COMMAND ----------

transaction_df =  spark.read.csv(transaction_file_path, inferSchema=True, header=True, mode= 'DROPMALFORMED')
identity_df = spark.read.csv(identity_file_path, inferSchema= True, header= True, mode= 'DROPMALFORMED')

# COMMAND ----------

print (f"Transaction dataset has {transaction_df.count()} rows and {len(transaction_df.columns)} columns")
print (f"Identity dataset has {identity_df.count()} rows and {len(identity_df.columns)} columns")

# COMMAND ----------

print("Transaction dataset columns:" ,transaction_df.columns)

# COMMAND ----------

print("Identity dataset columns", identity_df.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Join Trnsaction and Identity data farmes on TransactionID column

# COMMAND ----------

merged_df = transaction_df.join(identity_df,  transaction_df["TransactionID"] == identity_df["TransactionID"], how = 'inner').drop(identity_df["TransactionID"])
merged_records_count = merged_df.count()
print(f"No of columns = {len(merged_df.columns)}, No of Records = {merged_df.count()}")

# COMMAND ----------

# MAGIC %md 
# MAGIC #####Merged dataset has 144233 Records, 434 Columns

# COMMAND ----------

# MAGIC %md 
# MAGIC ####Summary table for Merged dataset

# COMMAND ----------

display(merged_df.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ###Aggregations and Visualizations

# COMMAND ----------

fraud_counts_df = merged_df.groupBy("isFraud").count()
display(fraud_counts_df[["isFraud", "count"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ####Fraud transaction percentages for diffrennt purchased product categories
# MAGIC ######Below stacked barchart indicates the percentage of Fraud transactions for different product categories,
# MAGIC ######With every transaction, there is purchased product category is available in the dataset. 

# COMMAND ----------

product_purchase = merged_df.groupBy("ProductCD").count()
product_purchase_fraud = merged_df.filter(merged_df["isFraud"] == 1).groupBy("ProductCD").count()
product_purchase_fraud = product_purchase_fraud.withColumnRenamed("count", "count_fraud")
product_purchase_fraud = product_purchase_fraud.join(product_purchase, product_purchase_fraud["ProductCD"] == product_purchase["ProductCD"]).drop( product_purchase["ProductCD"])
display(product_purchase_fraud)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Percentage of Fraud transactions by different  Credit and Debit card issuing companies.
# MAGIC Below chart shows the percentage of fraud transactions for cards issued by different comapanies.
# MAGIC The American Express issued cards have the lowest fraud transaction.
# MAGIC VISA issued cards have the highest percentage of fradu transactions.

# COMMAND ----------

transaction_card_type_df = merged_df.groupBy("card4").count()
fraud_transaction_card_type_df = merged_df.filter(merged_df["isFraud"] == 1).groupBy("card4").count()
fraud_transaction_card_type_df = fraud_transaction_card_type_df.withColumnRenamed("count", "count_fraud")
fraud_transaction_card_type_df = fraud_transaction_card_type_df.join(transaction_card_type_df, transaction_card_type_df["card4"] == fraud_transaction_card_type_df["card4"]).drop(transaction_card_type_df["card4"])
display(fraud_transaction_card_type_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##DATA Cleaning
# MAGIC 1) Only columns with atleast 95% NON-NULL values are considered for Analysis. <br>
# MAGIC 2) Drop records with missing values (from those with 95% complete data). <br>
# MAGIC 3) Create categorical variables from columns with string types\(Using StringIndexer). <br>
# MAGIC 4) Drop categorical columns with morethan 32 unique values. <br>

# COMMAND ----------

# MAGIC %md 
# MAGIC Calculate the percentage of Not Null values in each of the columns of merged data farme <br>
# MAGIC Calculated the proportion of values available for each of the columnd in the entire dataframe ([0 to 1]). <br>
# MAGIC Filtered columns for which 95% data is valid (Not Null) and saved into a new dataframe. <br>
# MAGIC From the new dataframe filter out records with missing details. <br>

# COMMAND ----------

not_null_result_df = merged_df.select([count(when(col(c).isNotNull() , c)).alias(c) for c in merged_df.columns])

# COMMAND ----------

print(not_null_result_df.columns,  len(not_null_result_df.columns))

# COMMAND ----------

col_list =  not_null_result_df.columns
val_list =  [val for val in not_null_result_df.collect()[0]]
zip_data = zip(col_list, val_list)
columns = ["Variable","Not_Null_Count"]
non_null_cnt_df = spark.createDataFrame(data= zip_data, schema= columns)
non_null_cnt_df.printSchema()
non_null_cnt_df = non_null_cnt_df.withColumn("Not_Null_Proportion", round(non_null_cnt_df["Not_Null_Count"]/merged_records_count, 2))
display(non_null_cnt_df)

# COMMAND ----------

complete_column_list = [column_row["Variable"] for column_row in  non_null_cnt_df.filter(non_null_cnt_df["Not_Null_Proportion"] ==1).select(["Variable"]).collect()]
print(f"Number of columns with no missing data are :{len(complete_column_list)}")

# COMMAND ----------

complete_95_column_list = [column_row["Variable"] for column_row in  non_null_cnt_df.filter(non_null_cnt_df["Not_Null_Proportion"] >=0.95).select(["Variable"]).collect()]

# COMMAND ----------

complete_at_95_df = merged_df[complete_95_column_list]
complete_at_95_df = complete_at_95_df.dropna()
print (f"Number of complete columns {complete_at_95_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Convert Categorical columns using StringIndexer and OneHotEncoding
# MAGIC 
# MAGIC Using StringIndexer convert columns with String factors into numercial data columns, there are 13 columns to be coverted.

# COMMAND ----------

complete_at_95_df = spark.read.csv(path = "dbfs:/mnt/complete_at_95_df.csv", inferSchema= True, header= True,mode= 'DROPMALFORMED')

# COMMAND ----------

## Code to create indexer columns
string_columns = [coltype[0] for coltype  in complete_at_95_df.dtypes if coltype[1] == 'string']
string_index_columns = [x+ "_index" for x in string_columns]
print(f"Cagegorical columns {string_columns}")

stages = []
for i in range(len(string_columns)):
  string_col = string_columns[i]
  indexer_col = string_index_columns[i]
  str_indexer = StringIndexer(inputCol= string_col, outputCol= indexer_col) 
  one_hot_encoder = OneHotEncoder(inputCol = str_indexer.getOutputCol(),  outputCol= str_indexer.getInputCol() + "classVec")
  stages.extend([str_indexer, one_hot_encoder])
one_hot_encoding = Pipeline(stages= stages)
complete_at_95_encoded_df = one_hot_encoding.fit(complete_at_95_df).transform(complete_at_95_df)    

# COMMAND ----------

complete_at_95_original_df = complete_at_95_encoded_df
display(complete_at_95_encoded_df.head(25))

# COMMAND ----------

##complete_at_95_encoded_df = complete_at_95_encoded_df_
###complete_at_95_df.write.option("header", "true").csv("dbfs:/mnt/complete_at_95_df.csv")

# COMMAND ----------

## From the dataframe drop the columns for which indexer columns were generated
## get a list of columns for which colnames starts with 'V', these columns are geneated by Vesta
##complete_at_95_df = complete_at_95_df.drop(*string_columns)
#complete_at_95_df = spark.read.csv(path = "dbfs:/mnt/complete_at_95_df.csv", inferSchema= True, header= True,mode= 'DROPMALFORMED')

# COMMAND ----------

# MAGIC %md
# MAGIC ##MODEL BUILDING AND EVALUATION
# MAGIC ##### M1 - DECISION TREE (With Fewre Predictors)
# MAGIC ##### M2- DECISION TREE 
# MAGIC ##### M2- LOGISTIC REGRESSION
# MAGIC ##### M3- GRADIENT BOOSTED TREE

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler,RFormula
from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder 

# COMMAND ----------

# MAGIC %md
# MAGIC #####SPLIT DATA INTO TRAINING AND VALIDATION SETS

# COMMAND ----------

#sub_complete_at_95_df = complete_at_95_df[sub_columns]
train_data, test_data  = complete_at_95_encoded_df.randomSplit([0.7, 0.3], 11)
print("Number of training records: " + str(train_data.count()))
print("Number of testing records : " + str(test_data.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC <b>
# MAGIC The MIN BIN  size for DECISION TREES nodes is 32, The DECISION TREE would throw an error if the MIN BIN size is less than the MAX NUMBER of Categories for a Feature.
# MAGIC To avoid the Max Bin error for DECISION TREE we've to set the MAX BIN to be miniimum as many as max number of categories for a feature <br>
# MAGIC <i> "IllegalArgumentException: requirement failed: DecisionTree requires maxBins (= 32) to be at least as large as the number of values in each categorical feature, but categorical feature 338 has 130 values. Consider removing this and other categorical features with a large number of values, or add more training examples. <i> <br>
# MAGIC <b>
# MAGIC Below code finds the Max number of categoreis in each of the category features.

# COMMAND ----------

# MAGIC %md
# MAGIC #####CREATE FEATURE COLUMNS LIST

# COMMAND ----------

## Take backup of the current dataframe, Create a list of columns to exclude from analysis (Original columns of onehot encoded columns)
column_list_at_95 = complete_at_95_encoded_df.columns
columns_to_exclude = []
columns_to_exclude.extend(string_columns)
columns_to_exclude.extend(string_index_columns)
columns_to_include = [col_name for col_name in complete_at_95_encoded_df.columns if col_name not in columns_to_exclude]
print (columns_to_include)

# COMMAND ----------

complete_at_95_encoded_df.count()

# COMMAND ----------

#index_col_list = [x for x in column_list_at_95 if x.endswith("index")]
categorical_features = string_columns
categorical_count ={}
for colname in categorical_features:
  distinct_count = complete_at_95_df.select(colname).distinct().count()
  categorical_count[colname] = distinct_count
    
categorical_col_cat_count = [col_name for col_name in categorical_count.items() if col_name[1] >= 32 ]
print(f"Categorical Features with high categories {categorical_col_cat_count}")

# COMMAND ----------

#feature_columns = [x for x in column_list_at_95 if not x.startswith("V")]
feature_columns = [x for x in columns_to_include]
feature_columns.remove("isFraud")
feature_columns.remove("TransactionID")
###feature_columns = [col_name for col_name in  feature_columns if not col_name in categorical_col_to_exclude]
print(feature_columns)

# COMMAND ----------

len(feature_columns)

# COMMAND ----------

# MAGIC %md
# MAGIC #####CREATE FORMULA, PIPELINE, PARAMGRIC OBJECTS

# COMMAND ----------

# RFormula transformation
formula = "{} ~ {}".format("isFraud", " + ".join(feature_columns))
rformula = RFormula(formula = formula)
print("Formula : {}".format(rformula))
pipeline_1 = Pipeline(stages=[]) 
basePipeline =[rformula]

# COMMAND ----------

# MAGIC %md
# MAGIC ##### DECISION TREE CLASSIFIER , LOGISTIC REGRESSION

# COMMAND ----------

### **** DECISION TREE CLASSIFIER ******
dt = DecisionTreeClassifier()
pl_dt = basePipeline + [dt]
# Parameter grid for the Decision Tree Classifier
pg_dt = (ParamGridBuilder()\
         .baseOn({pipeline_1.stages: pl_dt})\
         .addGrid(dt.maxBins,values= [135])
         .addGrid(dt.maxDepth, [3,5])\
         .build())

### **** LOGISTIC REGRESSIOn ******
lr = LogisticRegression()
pl_lr =  basePipeline + [lr]
pg_lr = (ParamGridBuilder()\
             .baseOn({pipeline_1.stages: pl_lr})\
             .addGrid(lr.regParam, [0.01, 0.5])\
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1 ])\
             .addGrid(lr.maxIter, [10, 15])\
             .build())

paramGrid = pg_lr + pg_dt 

# COMMAND ----------

# MAGIC %md
# MAGIC ##### TRAIN DECISIONTREE, LOGISTIC REGRESSION MODELS WITH FIVE FOLDS.

# COMMAND ----------

import mlflow
mlflow.sklearn.autolog()

# COMMAND ----------

with mlflow.start_run():
  evaluator = BinaryClassificationEvaluator()
  cv = CrossValidator()\
      .setEstimator(pipeline_1)\
      .setEvaluator(evaluator)\
      .setEstimatorParamMaps(paramGrid)\
      .setNumFolds(4)

  cvModel = cv.fit(train_data)

# COMMAND ----------

new_predictions = cvModel.transform(test_data)
import numpy as np 
print("Best Model")
print(cvModel.getEstimatorParamMaps()[np.argmax(cvModel.avgMetrics) ])
print("Worst Model")
print (cvModel.getEstimatorParamMaps()[np.argmin(cvModel.avgMetrics) ])

# COMMAND ----------

# MAGIC %md
# MAGIC ### MODEL  EVALUATION

# COMMAND ----------

# MAGIC %md
# MAGIC #### BEST MODEL EVALUATION

# COMMAND ----------

# Summarize the model over the training set and print out some metrics
Best_Model = cvModel.bestModel
Best_Model.save("dbfs:/mnt/Best_Model")
predictions_best = Best_Model_New.transform(test_data)

# COMMAND ----------

# MAGIC %md
# MAGIC #####GENERATE AUC, APR, ACCURACY, PRECISION, RECALL METRICS

# COMMAND ----------

from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
predict_labels_best_df = predictions_best.select("prediction", "label")
predict_labels_best_rdd =  predict_labels_best_df.rdd.map(list)

# COMMAND ----------

metrics_bc = BinaryClassificationMetrics(predict_labels_best_rdd)
print(f"Area under Average Precision (APR) curve for the Best Model is {metrics_bc.areaUnderPR}")
print(f"Area unde ROC curve for the Best Model is {metrics_bc.areaUnderROC}")

# COMMAND ----------

metrics_best_mc = MulticlassMetrics(predict_labels_best_rdd)
precision = metrics_best_mc.precision(1)
accuracy = metrics_best_mc.accuracy
recall = metrics_best_mc.recall(1)

# COMMAND ----------

recall = metrics_best_mc.recall(0)
precision = metrics_best_mc.precision(0)
print (recall, precision)

# COMMAND ----------

print(f"Precision of the Best Model is {precision}")
print(f"Accuarcy of the Best Model is {accuracy}")
print(f"Recall of the Best Model is {recall}")

# COMMAND ----------

# MAGIC %md
# MAGIC #####CONFUSION MATRIX FOR THE BEST MODEL

# COMMAND ----------

# MAGIC %md
# MAGIC <b>
# MAGIC 1) Area Under Precision-Recall (APR) Curve 0.5665381879629106 <br>
# MAGIC 2) The Areea Under curve (AUC) metric for the Decision Tree model is 0.6281324074853549<br>
# MAGIC 3) The Accuracy of the Decision Tree model is 0.9385708211400415 <br>
# MAGIC 4) The Precision of the Decision Tree model is 0.8532494758909853 <br>
# MAGIC 5) The Recall of the Decision Tree model is 0.2600638977635783 <br>
# MAGIC <b>

# COMMAND ----------

print (metrics_best_mc.confusionMatrix())

# COMMAND ----------

import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.plot([0, 1], [0, 1], 'r--')
plt.plot(Best_Model.stages[1].summary.roc.select('FPR').collect(),
         Best_Model.stages[1].summary.roc.select('TPR').collect())
plt.xlabel('False Positive Rate')  
plt.ylabel('True Positive Rate')
plt.show()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### M3 GRADIENT BOOSTED TREE

# COMMAND ----------

gbt = GBTClassifier()
gbt_rf =  basePipeline + [gbt]
gbparamGrid = (ParamGridBuilder()\
               .baseOn({pipeline_2.stages: gbt_rf})\
               .addGrid(gbt.maxDepth, [3, 5])\
               .addGrid(gbt.maxBins, [150,200])\
               .addGrid(gbt.maxIter, [10, 20])\
               .build())

paramGrid = gbparamGrid


# COMMAND ----------

mlflow.sklearn.autolog()
with mlflow.start_run():
  evaluator = BinaryClassificationEvaluator()
  cv = CrossValidator()\
  .setEstimator(pipeline_2)\
  .setEstimatorParamMaps(paramGrid)\
  .setEvaluator(evaluator)\
  .setNumFolds(4)

cvModel = cv.fit(train_data)

# COMMAND ----------

gbt_predictions = cvModel.transform(test_data)
predict_labels_df = gbt_predictions.select("prediction", "label")
predict_labels_rdd =  predict_labels_df.rdd.map(list)

# COMMAND ----------

gbtmetrics_bc = BinaryClassificationMetrics(predict_labels_rdd)
gbtmetrics_mc = MulticlassMetrics(predict_labels_rdd)

# COMMAND ----------

gbt_areaUnderPR = gbtmetrics_bc.areaUnderPR
gbt_areaUnderROC = gbtmetrics_bc.areaUnderROC
gbt_accuracy =  gbtmetrics_mc.accuracy
gbt_precision  =  gbtmetrics_mc.precision(1)
gbt_recall = gbtmetrics_mc.recall(1)
gbt_precision_0  =  gbtmetrics_mc.precision(0)
gbt_recall_0 = gbtmetrics_mc.recall(0)
print(gbt_precision_0, gbt_recall_0)

# COMMAND ----------

print (gbtmetrics_mc.confusionMatrix().toArray())

# COMMAND ----------

# MAGIC %md
# MAGIC For Gradent Boosted Tree model <br>
# MAGIC 1) Area Under Precision-Recall (APR) Curve is 0.8608608608608609 <br>
# MAGIC 2) The Areea Under curve (AUC) metric for the Decision Tree model 0.7709884368993762 <br>
# MAGIC 3) The Accuracy of the Decision Tree model is 0.9577799454741002 <br>
# MAGIC 4) The Precision of the Decision Tree model is 0.8608608608608609 <br>
# MAGIC 5) The Recall of the Decision Tree model is 0.549520766773163<br>

# COMMAND ----------

# MAGIC  %md
# MAGIC  ###MODELS EVALUATION (M1 - DECISION TREE with fewer predictors)

# COMMAND ----------

# MAGIC %md
# MAGIC #####Predict the classification for test data.

# COMMAND ----------

predictions = cvModel.transform(test_data)

# COMMAND ----------

# MAGIC %md
# MAGIC #####EVALUATE THE MODEL USING ACCURACY, RECALL AND AREA UNDER CURVE METRICS

# COMMAND ----------

# MAGIC %md
# MAGIC Since this is an imbalanced dataset (8% Fraud rate in the merged dataset), The recall value is more importnat than the Accuracy <br>
# MAGIC Area udner Precision-Recall Curve, Area Under Curve (AUC), Accuracy, Recall values are measured.<br>
# MAGIC The AUC = 0.663 and Area Under PR = 0.3957 indicate, that the model performs better than the Random model<br>
# MAGIC Accuracy at ~80% indicates the model can be still imporved.<br>

# COMMAND ----------

evaluator_new = BinaryClassificationEvaluator(labelCol="isFraud", rawPredictionCol="prediction")

# COMMAND ----------

AUPR_Value = (evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"}))
print("Area under PR = %s" % AUPR_Value)

# COMMAND ----------

AUC_Value = (evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"}))
print(AUC_Value)

# COMMAND ----------

accuracy = evaluator_new.evaluate(predictions, {evaluator.metricName: "accuracy"})

# COMMAND ----------

recall = evaluator_new.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
print(recall)

# COMMAND ----------

# MAGIC %md
# MAGIC <b>
# MAGIC 1) Area under Precision-Recall Curve 0.39576 <br>
# MAGIC 2) The area under curve (AUC) metric for the Decision Tree model is 0.663 <br>
# MAGIC 3) The Accuracy of the Decision Tree model is 0.79<br>
# MAGIC 4) The Weighted recall of the model is 0.791<br>
# MAGIC <b>

# COMMAND ----------

# MAGIC %md 
# MAGIC Below table shows the recoreds from test data where the prediction has not matched with the actual label.

# COMMAND ----------

display(predictions[predictions["isFraud"] != predictions["prediction"]])
