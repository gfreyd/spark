package com.sparkProject

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}

object Job {

  def main(args: Array[String]): Unit = {

    // SparkSession configuration
    val spark = SparkSession
      .builder
      .appName("spark session TP_parisTech")
      .getOrCreate()

    val sc = spark.sparkContext

    import spark.implicits._


    /********************************************************************************
      *
      *        TP 1
      *
      *        - Set environment, InteliJ, submit jobs to Spark
      *        - Load local unstructured data
      *        - Word count , Map Reduce
      ********************************************************************************/



    // ----------------- word count ------------------------

    //val df_wordCount = sc.textFile("/users/maxime/spark-1.6.2-bin-hadoop2.6/README.md")
      val df_wordCount = sc.textFile("/cal/homes/gfreyd/spark-2.0.0-bin-hadoop2.6/README.md")
      .flatMap{case (line: String) => line.split(" ")}
      .map{case (word: String) => (word, 1)}
      .reduceByKey{case (i: Int, j: Int) => i + j}
      .toDF("word", "count")

    df_wordCount.orderBy($"count".desc).show()


    /********************************************************************************
      *
      *        TP 2 : début du projet
      *
      ********************************************************************************/
    //val cars = sqlContext.csvFile("/cal/homes/gfreyd/INF729/Spark/TP2-3/cumulative.csv")

    //Question 3.a
    val path = "/cal/homes/gfreyd/INF729/Spark/TP2-3/cumulative.csv"
    val df = spark.read.option("header","true")
      .option("separator",",")
      .option("comment","#")
      .option("inferSchema", "true") // Automatically infer data types
      .csv(path)

    //Question 3.b
    println("number of columns", df.columns.length)
    println("number of rows", df.count)

    //Question 3.c
    df.show()

    //Question 3.d
    val columns = df.columns.slice(10, 20) // select first 10 columns
    df.select(columns.map(col): _*).show(50) // show first 10 columns

    //Question 3.e
    df.printSchema()

    //Question 3.f
    df.groupBy($"koi_disposition").count().show()

    //Question 4.a
    val df_cleaned =  df.filter($"koi_disposition" === "CONFIRMED" || $"koi_disposition" === "FALSE POSITIVE")
    //other solution val filterDf = df.filter("koi_disposition in (\"CONFIRMED\", \"FALSE POSITIVE\")")

    //Question 4.b
    df_cleaned.groupBy($"koi_eccen_err1").count().show()

    //Question 4.c
    val df_cleaned2 = df_cleaned.drop($"koi_eccen_err1")

    //Question 4.d
    val df_cleaned3 = df_cleaned2.drop("index","kepid","koi_fpflag_nt","koi_fpflag_ss","koi_fpflag_co","koi_fpflag_ec",
      "koi_sparprov","koi_trans_mod","koi_datalink_dvr","koi_datalink_dvs","koi_tce_delivname",
      "koi_parm_prov","koi_limbdark_mod","koi_fittype","koi_disp_prov","koi_comment","kepoi_name","kepler_name",
      "koi_vet_date","koi_pdisposition")

    //Question 4.e
    val useless = for(col <- df_cleaned3.columns if df_cleaned3.select(col).distinct().count() <= 1 ) yield col
    val df_cleaned4 = df_cleaned3.drop(useless: _*)

    //Question 4.f
    //df_cleaned4.describe().show()
    df_cleaned4.describe("koi_impact", "koi_duration").show()

    //val summary = Statistics.colStats(df3.columns)
    //print(summary)

    //MultivariateStatisticalSummary summary = Statistics.colStats(mat.rdd());
    //System.out.println(summary.mean());

    //Question 4.g
    val df_filled = df_cleaned4.na.fill(0.0)

    //Question 5
    val df_labels = df_filled.select("rowid", "koi_disposition")
    val df_features = df_filled.drop("koi_disposition")

    val df_joined = df_features
      .join(df_labels, usingColumn = "rowid")

    //Question 8.a
    def udf_sum = udf((col1: Double, col2: Double) => col1 + col2)


    val df_newFeatures = df_joined
      .withColumn("koi_ror_min", udf_sum($"koi_ror", $"koi_ror_err2"))
      .withColumn("koi_ror_max", $"koi_ror" + $"koi_ror_err1")

    //Question 9
    df_newFeatures
      .coalesce(1) // optional : regroup all data in ONE partition, so that results are printed in ONE file
      // >>>> You should not do that in general, only when the data are small enough to fit in the memory of a single machine.
      .write
      .mode("overwrite")
      .option("header", "true")
      .csv("/cal/homes/gfreyd/INF729/Spark/TP2-3/cleanedDataFrame.csv")
  }
}
