import scala.io.Source
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import java.io.File

class JoinFiles {
  def main(path: String): Unit = {

    val conf = new SparkConf()
      .setAppName("Join Files ")
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

  

    val filesFil = mainDir.listFiles.filter(_.isDirectory())
      .flatMap(l => {
        l.listFiles().filter(_.getName.endsWith(".csv"))
      })

    val datosDF = filesFil.map(f => {
      val brca = spark.read.option("header", "true").option("inferSchema", "true").option("sep", ",").
        option("maxColumns", 400000).csv(f.getAbsolutePath)
      val res1 = brca.columns.filter { (colName: String) =>
        brca.filter(brca(colName).isNull).isEmpty
      }.toList
      val brcares = brca.select(res1.head, res1.tail: _*)
      brcares.write.option("header", "true").csv(dirName.concat("noNA/").concat(f.getName))
    })

     val mainDir2 = new java.io.File(dirName.concat("noNA"))

     val filesFil2 = mainDir2.listFiles.filter(_.isDirectory())
      .flatMap(l => {
        l.listFiles().filter(_.getName.endsWith(".csv"))
      })
      
    val data = filesFil2.map(f => {
      (Source.fromFile(f.getAbsolutePath).
        getLines()).toArray
    })

    val pathClassFollowUP = "./data/manifestFollowUPOutput/classFollowUP.txt"
    val classFollowUP = Source.fromFile(pathClassFollowUP).getLines().map(s =>
      {
        val cad = s.split(",")
        (cad(0), cad(1))
      }).toMap

    val nData = data.length
    println(nData)
    println(data.apply(0).length)

    val array = new Array[String](data.apply(0).length)

  

    for (i <- 0 to nData - 1) {
      for (j <- 0 to data.apply(0).length - 1) {
        val line = data.apply(i).apply(j)
        var vcase = line.split(",")
        val kcase = vcase(0)
      

        val value = array.apply(j)

        //        println("vcase " + vcase.mkString(" "))

        if (value != null) {
       
          val x = value.concat(",").concat(vcase.drop(1).mkString(","));
       

          array.update(j, x)
        } else {
          val x = vcase.drop(1).mkString(",")
        
          val classFU = classFollowUP.get(kcase)
          if (classFU != None) {
            array.update(j, kcase.concat(",").concat(classFU.get).concat(",").concat(x))
          } else {
            array.update(j, kcase.concat(",").concat("class").concat(",").concat(x))
          }
        }
      }
    }

    var brcaFile = new File("brcaOut.csv");
    printToFile(brcaFile) {

      p =>
        {
          array.foreach(x => p.println(x));

        }
    }
    sc.stop()

  }

  def printToFile(f: java.io.File)(op: java.io.PrintWriter => Unit) {
    val p = new java.io.PrintWriter(f)
    try { op(p) } finally { p.close() }
  }
}