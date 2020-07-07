import scala.io.Source
import java.io._
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.HashPartitioner
import org.apache.spark.RangePartitioner
import org.apache.spark.sql.Row

class GenerateFile {
  def principal(path: String): Unit = {

    val conf = new SparkConf()
      .setAppName("generateFile")
      .setMaster("local[*]")

    val sc = new SparkContext(conf)

    val spark = SparkSession
      .builder().master("local[2]")
      .appName("correlation")
      .config("spark.network.timeout", "10000000")
      .config("spark.executor.heartbeatInterval", "10000000")
      .config("spark.sql.pivotMaxValues", "500000")
      .config("spark.driver.memory", "16G")
      .config("spark.executor.memory", "16G")
      .config("spark.worker.memory", "16G")
      .getOrCreate()

    import spark.implicits._

    
    val dirName = path

    val mainDir = new java.io.File(dirName)
    val filteredFiles = sc.parallelize(mainDir.listFiles.filter(_.isDirectory()).
      flatMap(l => {
        (l.listFiles().filter(_.getName.endsWith("hg38.txt")).
          filter(_.getName.startsWith("j"))
          .filter(_.getName.contains("01A-11D")))

      }))

  

    var data = (filteredFiles.flatMap(f => {
      (Source.fromFile(f.getAbsolutePath).
        getLines().drop(1).map(s => s.split("	")).
        map(s => {
          val pos = f.getName.indexOf("TCGA")
          (f.getName.substring(pos, pos + 12), s(0), s(1))
        }))
    })).toDF("case", "composite", "value").cache()
    
    
   var dataNa = data.filter($"value" === "NA").select("composite").distinct().map(_.get(0).asInstanceOf[String]).collect()
  
    
    data = data.filter(not(data.col("composite").isin(dataNa:_*)))


    
    import org.apache.spark.sql.functions.{ array, collect_list }

    val valueIds = collect_list(($"value")).alias("value")
    val caseIds = collect_list(($"case")).alias("case")

    val npartitions = 500
    val nElements = 1000
    var npart = 0
    var cpgsiteDist = data.select("composite").distinct().
      map(_.get(0).asInstanceOf[String]).cache()

    while (cpgsiteDist.count() > 0) {
      println("partition " + npart)
      npart = npart + 1;
      val cpgsiteDistArray = cpgsiteDist.take(nElements)

      val data2 = data.filter(l => cpgsiteDistArray.contains(l.get(1))).cache()
      val dataGrouped = data2.groupBy($"case").pivot("composite").
        agg(min($"value")).cache()

         val res1 = dataGrouped.columns.filter { (colName: String) =>
        dataGrouped.filter(dataGrouped(colName).isNull).count() == 0
      }.toList
      val dataGroupedRed = dataGrouped.select(res1.head, res1.tail: _*).cache()

          dataGroupedRed.coalesce(1).write.option("header", "true").csv("./dataTemp/data".concat(npart.toString).concat(".csv"))

      cpgsiteDist = cpgsiteDist.filter(l => !cpgsiteDistArray.contains(l)).cache()
    }
    sc.stop
  }
}