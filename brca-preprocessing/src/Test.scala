
import scala.io.Source
import java.io._
import org.apache.log4j.Level
import org.apache.log4j.Logger

object Test {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val filterData = new FilterData()
    filterData.main()

    val generateFile = new GenerateFile()
    val pathPatients: String = "./data/patients"
    generateFile.principal(pathPatients);

    val pathTemp = ("./dataTemp")
    val joinFiles = new JoinFiles();
    joinFiles.main(pathTemp)
 

  }

}