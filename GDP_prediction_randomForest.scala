// Databricks notebook source
import org.apache.spark.ml.{PipelineModel, Pipeline}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.sql.DataFrameNaFunctions
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.Imputer
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, CrossValidatorModel}
import org.apache.spark.ml.feature.{PCAModel, PCA, Normalizer}
import org.apache.spark.ml.regression.LinearRegression
import scala.collection.mutable.HashMap
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.expressions.Window

// Import data.
var headlines = spark.table("millenniumofdata_v3_headlines_csv")


// Filter out the rows contain invalid labels.
headlines = headlines.na.fill(0).filter($"Real GDP of England at market prices" > 0)
// Change target column name to label.
headlines = headlines.withColumnRenamed("Real GDP of England at market prices", "label")
// Remove the '.' from the column names.
headlines = headlines.withColumnRenamed("Nominal UK GDP at market prices.1", "Nominal UK GDP at market prices1")
                     .withColumnRenamed("Labour productivity.1", "Labour productivity1")
                     .withColumnRenamed("Bank Rate.1", "Bank Rate1")
                     .withColumnRenamed("Bank of England Balance sheet.1", "Bank of England Balance sheet1")
                     .withColumnRenamed("Public sector Total Managed Expenditure.1", "Public sector Total Managed Expenditure1")
                     .withColumnRenamed("Public Sector Net Lending(+)/Borrowing(-).1", "Public Sector Net Lending(+)/Borrowing(-)1")
                     .withColumnRenamed("Central Government Gross Debt.1", "Central Government Gross Debt1")
                     .withColumnRenamed("Trade deficit.1", "Trade deficit1")
                     .withColumnRenamed("Current account.1", "Current account1")
                     .withColumnRenamed("Current account deficit including estimated non-monetary bullion flows.1", "Current account deficit including estimated non-monetary bullion flows1")
                     .withColumnRenamed("Public Sector Total Receipts.1", "Public Sector Total Receipts1")
                     .withColumnRenamed("UK Public sector debt.1", "UK Public sector debt1")
                     .withColumnRenamed("UK Public sector debt.2", "UK Public sector debt2")
                     .withColumnRenamed("Current account .1", "Current account1")
// Assign id to column.
headlines = headlines.withColumn("id", monotonically_increasing_id())

// Add last year GDP as a feature of current year
val w = Window.orderBy($"id")
headlines = headlines.withColumn("Last year real GDP of England at market prices", lag($"label", 1).over(w))
headlines.select("id", "label", "Last year real GDP of England at market prices").show(30)

// Use imputer to set nan values to the mean of the corresponding columns.
val cols = headlines.schema.names
// Convert all columns to Double.
for (cols <-cols ){ headlines = headlines.withColumn(cols, col(cols).cast("Double"))}
// Set the null values to the mean of columns.
val imputer = new Imputer().
setInputCols(cols).
setOutputCols(cols.map(c => c)).
setMissingValue(0).
setStrategy("mean")
headlines = imputer.fit(headlines).transform(headlines)


// COMMAND ----------

// Data preprocessing and data cleaning: remove the GDP related features.
val cols_to_use = (ArrayBuffer(headlines.schema.names: _*) --= Array(
                                                 "label",
                                                 "Real GDP of England at factor cost ", 
                                                 "Real UK GDP at factor cost_ geographically-consistent estimate based on post-1922 borders", 
                                                 "Real UK GDP at market prices_ geographically-consistent estimate based on post-1922 borders", 
                                                 "id",
                                                 "Nominal GDP of England at market prices",
                                                 "Nominal UK GDP at market prices",
                                                 "Nominal UK GDP at market prices1"
                                                )).toArray

// Use VectorAssembler to put all features into one vector, outputing the vector to "features" column.
val assembler = new VectorAssembler()
                    .setInputCols(cols_to_use)
                    .setOutputCol("features")

// Use normalizer to normalize all features, outputing the result to "normalized_feature".
val normalizer = new Normalizer()
                 .setInputCol("features")
                 .setOutputCol("normalized_features")

// Use the data from 1270 to 2010 for training
val total_row = headlines.count()
val train_up_to = 2010
val training = headlines.filter($"Description" <= train_up_to)

// COMMAND ----------

// Build my own rolling k fold validator 
class rollingKFoldValidator(val estimator: Pipeline, val estimatorParamMaps: Array[ParamMap], val numFold: Int, val evaluator: Evaluator){
  val trainingRatio = 0.85;
  var errorMap: HashMap[Int, Double] = new HashMap() 
  
  // Split dataset into training folds and testing folds
  def split_train_test(dataset: Dataset[_], currentK: Int): (Dataset[_], Dataset[_]) = {
    val total_row = dataset.count()
    val numEachFold = (total_row / this.numFold).toInt
    val trainingSize = ((currentK+1) * numEachFold * trainingRatio).toInt
    val training = dataset.filter($"id" <= trainingSize)
    val testing = dataset.filter($"id" > trainingSize).filter($"id" < ((currentK+1)*numEachFold))
    return (training, testing)
  }
  
  // Train the model by combining rolling k fold with param grid search
  def fit(dataset: Dataset[_]): PipelineModel = {
    for( currentK <- 0 until this.numFold) {
         var (training, testing) = split_train_test(dataset, currentK)
         // Returns a sequence of estimators. Each estimators is fitted by a particular param map
         var estimators: Seq[PipelineModel] = this.estimator.fit(dataset, this.estimatorParamMaps)
         // Calculate the errors for each estimator and accumulate the errors by strong them into the hashmap
         for (estimator <- estimators) {
           var paramGridIndex = 0
           var predictions = estimator.transform(testing)
           var error: Double = this.evaluator.evaluate(predictions)
           if (currentK == 0){
             this.errorMap = this.errorMap+=(paramGridIndex -> error)
           } else {
             this.errorMap(paramGridIndex) = errorMap(paramGridIndex)+error
           }
           paramGridIndex+=1
         }
        
    }
    
    // train a new estimator using the best param map
    // Get the best param map by the least error
    val (bestParamMapIndex, bestError) = this.errorMap.minBy { case (key, value) => value }
    val bestParamMap = this.estimatorParamMaps(bestParamMapIndex)
    val finalEstimator = this.estimator.fit(dataset, bestParamMap)
    return finalEstimator
  }
}

// COMMAND ----------

import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param.shared.{HasLabelCol, HasPredictionCol, HasWeightCol}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{Dataset, functions => F}


class SMapeEvaluator(override val uid: String) extends Evaluator with HasPredictionCol with HasLabelCol  {

    def this() = this(Identifiable.randomUID("wrmseEval"))

    def setPredictionCol(value: String): this.type = set(predictionCol, value)
    
    def setLabelCol(value: String): this.type = set(labelCol, value)
    
    def evaluate(dataset: Dataset[_]): Double = {
        dataset
            .withColumn("residual", F.abs(F.col(getLabelCol) - F.col(getPredictionCol)))
            .withColumn("absSum", (F.abs(F.col(getLabelCol)) + F.abs(F.col(getPredictionCol)))/2)
            .select(
                (F.sum(F.col("residual")/F.col("absSum"))*100)/F.count(F.col("residual"))
            )
            .collect()(0)(0).asInstanceOf[Double]

    }

    override def copy(extra: ParamMap): Evaluator = defaultCopy(extra)

    override def isLargerBetter: Boolean = false
}

// COMMAND ----------


// Train a RandomForest model.
val rf = new RandomForestRegressor()
  .setLabelCol("label")
  .setFeaturesCol("normalized_features")

// Pipeline for the training
val pipeline = new Pipeline()
  .setStages(Array(assembler, normalizer, rf))

// Use grid search to search for the best hyperparameters.
val paramGrid = new ParamGridBuilder()
  .addGrid(rf.maxDepth, Array(10, 20, 30))
  .addGrid(rf.numTrees, Array(10, 20, 40, 80))
  .addGrid(rf.maxBins, Array(32, 48))
  .build()


val evaluator = new SMapeEvaluator()
                 .setLabelCol("label")
                 .setPredictionCol("prediction")
          

// We now treat the Pipeline as an Estimator, wrapping it in a rollingKFoldValidator instance.
// This will allow us to jointly choose parameters for all Pipeline stages.
val rv = new rollingKFoldValidator(pipeline, paramGrid, 4, evaluator)

// Run rolling k-fold validation, and choose the best set of parameters.
val rvModel = rv.fit(training)

// Calculate train error for comparison
val prediction_train = rvModel.transform(training)
prediction_train.select("prediction", "label", "normalized_features").show(5)
val smape_train = evaluator.evaluate(prediction_train)
println(s"Symmetric mean absolute percentage error (SMAPE) on train data = $smape_train")

// COMMAND ----------

// prediction over 1 year
val testing = headlines.filter($"Description" === 2011)
// Make predictions.
var predictions = rvModel.transform(testing)
predictions.select("prediction", "label", "normalized_features").show(5)
val smape = evaluator.evaluate(predictions)
println(s"Symmetric mean absolute percentage error (SMAPE) over 1 year = $smape")

// COMMAND ----------

// prediction over 2 years
val testing = headlines.filter($"Description" === 2011 || $"Description" === 2012)
// Make predictions.
var predictions = rvModel.transform(testing)
predictions.select("prediction", "label", "normalized_features").show(5)
val smape = evaluator.evaluate(predictions)
println(s"Symmetric mean absolute percentage error (SMAPE) over 2 years = $smape")

// COMMAND ----------

// prediction over 3 years
val testing = headlines.filter($"Description" === 2011 || $"Description" === 2012 || $"Description" === 2013)
// Make predictions.
var predictions = rvModel.transform(testing)
predictions.select("prediction", "label", "normalized_features").show(5)
val smape = evaluator.evaluate(predictions)
println(s"Symmetric mean absolute percentage error (SMAPE) over 3 years = $smape")

// COMMAND ----------

// prediction over 6 years
val testing = headlines.filter($"Description" > train_up_to)
// Make predictions.
var predictions = rvModel.transform(testing)
predictions.select("prediction", "label", "normalized_features").show(5)
val smape = evaluator.evaluate(predictions)
println(s"Symmetric mean absolute percentage error (SMAPE) over 6 years = $smape")

// COMMAND ----------

// Display our testing result by tables and graphs
display(predictions.select("prediction", "label", "normalized_features", "Description"))

// COMMAND ----------

// Display the percentage error of each GDP year prediction
predictions = predictions.withColumn("Percentage error", abs(($"prediction"-$"label")/$"label"))
display(predictions.select("Percentage error", "Description"))

// COMMAND ----------

// The best hyperparameters for the RandomForestRegressorModel
rvModel.asInstanceOf[PipelineModel].stages(2).asInstanceOf[RandomForestRegressionModel].extractParamMap()

// COMMAND ----------

val importanceVector = rvModel.asInstanceOf[PipelineModel].stages(2).asInstanceOf[RandomForestRegressionModel].featureImportances
// Display the feature importance with index
var featureImportances: List[(Int,Double)] = List()
importanceVector.toArray.zipWithIndex
            .map(_.swap)
            .sortBy(-_._2)
            .foreach(x => featureImportances = featureImportances:+((x._1, x._2*100)))

val df = featureImportances.toDF("feature index", "feature importances")
display(df.select("feature index", "feature importances"))

// COMMAND ----------

// plot the feature importance with feature names
var importanceWithNames: List[(String, Double)] = List()
featureImportances.foreach(x => importanceWithNames = importanceWithNames:+((cols_to_use(x._1), x._2)))
val dfWithNames = importanceWithNames.toDF("feature name", "feature importances")
display(dfWithNames.select("feature name", "feature importances"))
