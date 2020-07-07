
import scala.io.Source
import java.io._

class FilterData {
  def main(): Unit = {
    //Paths
    val fileFollowUp = "./data/manifestFollowUPInput/nationwidechildrens.org_clinical_follow_up_v4.0_brca.txt";
    val fileManifest = "./data/manifestFollowUPInput/gdc_manifest_20190410_100216.txt"; // Metilation 450
    val classpathOutput = "./data/manifestFollowUPOutput/"

    var followUP = (processFollowUp(fileFollowUp))
    followUP = filterAliveDeadFollowUP(followUP._1, followUP._2)

    var CASEID = processManifest(fileManifest);
    CASEID = filterManifest(CASEID._1, CASEID._2)

    val CASEIDFilteredFollowUP = matchingFollowUpManifest(CASEID, followUP)

    val CASEIDFiltered = CASEIDFilteredFollowUP._1
    val followUPFiltered = CASEIDFilteredFollowUP._2

    println(CASEIDFiltered.length)

    println(followUPFiltered.length)

    //Count YES
    println(followUPFiltered.filter(p => p(7).equals("YES")).length)

    //Count NO
    println(followUPFiltered.filter(p => p(7).equals("NO")).length)
    followUPFiltered.foreach(s => println(s.mkString("	")))

    // Print filtered followUP 
    var followUPFile = new File(classpathOutput + "followUP.txt");
    printToFile(followUPFile) {

      p =>
        {
          followUP._1.foreach(x => p.println(x.mkString("	")));
          followUPFiltered.foreach(x => p.println(x.mkString("	")))
        }
    }
    
    val mapCase = collection.mutable.Map[String, String]()
    followUPFiltered.foreach(x => {
      val valueOld = mapCase.get(x(1))
      val valueNew = x(7)
     
      if (valueOld!=None) {
        if (valueOld.get.equals("YES") || valueNew.equals("YES")){
          mapCase.put(x(1),"YES")
        } else 
          mapCase.put(x(1),valueNew)
      } else 
      mapCase.put(x(1),valueNew)
      })
      
 
      var classFollowUPFile = new File(classpathOutput + "classFollowUP.txt");
      printToFile(classFollowUPFile) {

      p =>
        {
          mapCase.foreach(x => p.println(x._1+","+x._2))
        }
       }

    // Print filtered manifest including CASE

    var manifestFileCASE = new File(classpathOutput + "manifestCASE.txt");
    printToFile(manifestFileCASE) {

      p =>
        {
          p.print("CASE" + "	")
          CASEID._1.foreach(x => p.println(x));
          CASEIDFiltered.foreach(x => p.println(x))
        }
    }

     // Print filtered manifest without CASE

    var manifestFile = new File(classpathOutput + "manifest.txt");
    printToFile(manifestFile) {

      p =>
        {
          CASEID._1.foreach(x => p.println(x));
          CASEIDFiltered.map(x => x.split("	").drop(1).mkString("	")).foreach(x => p.println(x))
        }
    }

  }

  def processFollowUp(file: String): (Array[Array[String]], Array[Array[String]]) = {

    val lines = Source.fromFile(file).getLines().map(s => s.split("	"))

    //Drop lines not used
    val linesFiltered1 = lines.map(s => (Array(s(0), s(1), s(4), s(8), s(9), s(10), s(11), s(12))))

    //Column names
    val columns = linesFiltered1.drop(1).take(1).toArray;

    //Data
    val rows = linesFiltered1.drop(1).toArray;


    columns.foreach(s => println(s.mkString(" ")))

    (columns, rows)
  }

  def filterAliveDeadFollowUP(columns: Array[Array[String]], rows: Array[Array[String]]): (Array[Array[String]], Array[Array[String]]) = {
    /*
    0 -> bcr_patient_uuid
    1 -> bcr_patient_barcode
    2 -> form_completion_date
    3 -> person_neoplasm_cancer_status
    4 -> vital_status
    5 -> days_to_last_followup
    6 -> days_to_death
    7 -> new_tumor_event_after_initial_treatment
    */

    // filtered alive or dead with more than 5 years of treatment
    
    val rowsFil = rows.filter(p => (
      p(7).equals("YES")
      || (p(7).equals("NO") &&
        ((p(4).equals("Alive") && p(5).toInt > 1825)
          || (p(4).equals("Dead") && p(6).toInt > 1825)))))

    (columns, rowsFil)
  }

  def processManifest(file: String): (Array[String], Array[String]) = {
    val l = Source.fromFile(file).getLines()

    val columns = l.take(1).toArray
    val lines = l.drop(1).map(s => s.split("	")).toArray

    val CASEID = lines.map(s => {
      val index = s(1).indexOf("TCGA")
      if (index != -1) {
        val cad = s(1).substring(index, index + 12);
        cad + "	" + s(0) + "	" + s(1) + "	" + s(2) + "	" + s(3) + "	" + s(4)
      } else ""
    })

    (columns, CASEID)

  }

  def filterManifest(columns: Array[String], rows: Array[String]) = {
    val CASEID = rows.filter(l => {
      val s = l.split("	")

      val index = s(2).indexOf("TCGA")
      if (index != -1) {
        val cad = s(2).substring(index, index + 28);
        val cadSplitted = cad.split("-") //Pos 3 -> Sample type, Pos 4 ->Portion
        (cadSplitted(3).equals("01A") && cadSplitted(4).equals("11D"))
        // (cad + "	" + s(0) + "	" + s(1) + "	" + s(2) + "	" + s(3) + "	" + s(4))

      } else false
    })

    (columns, CASEID)
  }

  def matchingFollowUpManifest(CASEID1: (Array[String], Array[String]), followUp1: (Array[Array[String]], Array[Array[String]])): (Array[String], Array[Array[String]]) = {

    val CASEID = CASEID1._2
    val followUp = followUp1._2

    val names = followUp.map(p => p(1))
    val CASEIDFiltrado = CASEID.filter(p => names.contains(p.split("	")(0)))
    
    //New
    val namesCASE = CASEIDFiltrado.map(p => p.split("	")(0))

    CASEIDFiltrado.foreach(s => println(s))

    val followUPfilter = followUp.filter(p => namesCASE.contains(p(1)))

    (CASEIDFiltrado, followUPfilter)
  }

  def printToFile(f: java.io.File)(op: java.io.PrintWriter => Unit) {
    val p = new java.io.PrintWriter(f)
    try { op(p) } finally { p.close() }
  }
}