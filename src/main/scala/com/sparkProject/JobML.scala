package com.sparkProject

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.tuning.TrainValidationSplit
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

/**
  * Exoplanets classifier to predict their suitabibility for extraterrestrial life
  * Gregory FREYD et Clément BEGOTTO
  */
object JobML {

  def main(args: Array[String]): Unit = {

    // SparkSession configuration
    val spark = SparkSession
      .builder
      .appName("spark session TP_parisTech")
      .getOrCreate()

    val sc = spark.sparkContext

    /************************ PATH SECTION ******************************/
    val path = "/cal/homes/gfreyd/INF729/Spark/TP2-3/cleanedDataFrame.csv"
    val modelOutputPath = "/cal/homes/gfreyd/INF729/Spark/TP2-3/model"
    /************************ PATH SECTION ******************************/


    val df = spark.read.option("header", "true")
      .option("separator", ",")
      .option("comment", "#")
      .option("inferSchema", "true") // Automatically infer data types
      .csv(path)

    println("number of columns", df.columns.length)
    println("number of rows", df.count)

    //Question 1.a
    val columnsArray = df.columns.filter(_ != "rowid").filter(_ != "koi_disposition")
    println("number of columns", columnsArray.length)

    val assembler = new VectorAssembler()
      .setInputCols(columnsArray)
      .setOutputCol("features")

    val dfTransformed = assembler.transform(df)
    println(dfTransformed.select("features").show(10))

    //Old Question 1.b
    //We do not use StandardScaler because it converts vectors in sparse vectors which are not handled properly
    //after transformation by other Spark objects
    //Data will be rescaled using the method provided in the model

    //Question 1.b
    val indexer = new StringIndexer()
      .setInputCol("koi_disposition")
      .setOutputCol("label")

    val indexed = indexer.fit(dfTransformed).transform(dfTransformed)

    println(indexed.select("label").show(10))

    //Question 2.a
    val Array(training, test) = indexed.randomSplit(Array(0.9, 0.1))

    val Array(training_set, validation_set) = training.randomSplit(Array(0.7, 0.3))
    println(training_set.show())

    //Question 2.b
    val lr = new LogisticRegression()
      .setElasticNetParam(1.0)  // L1-norm regularization : LASSO
      .setLabelCol("label")
      .setStandardization(true)  // to scale each feature of the model
      .setFitIntercept(true)  // we want an affine regression (with false, it is a linear regression)
      .setTol(1.0e-5)  // stop criterion of the algorithm based on its convergence
      .setMaxIter(300)  // a security stop criterion to avoid infinite loops


    // Fit the model
    val lrModel = lr.fit(training_set)
    val prediction = lrModel.transform(validation_set)

    // Print the coefficients and intercept for logistic regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    // Logaritmic scale, step 0.5
    val powersArray = -6.0 to (0.0, 0.5) toArray
    val logScaleParamsArray = powersArray.map(x => math.pow(10, x))
    println(logScaleParamsArray)

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, logScaleParamsArray)
      .build()

    // In this case the estimator is simply the linear regression.
    // A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(lr)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    // Run train validation split, and choose the best set of parameters.
    val model = trainValidationSplit.fit(training)

    // Make predictions on test data. model is the model with combination of parameters
    // that performed best.
    val predictionWithGrid = model.transform(test)
      .select("features", "label", "prediction")

    predictionWithGrid.show()

    // Compute model accuracy against results on test set
    val evaluator = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("prediction")
    val accuracy = evaluator.evaluate(predictionWithGrid)
    println(s"*** Model accuracy is: ${accuracy}")

    // Save model to disk
    model.write.overwrite.save(modelOutputPath)
  }
}
